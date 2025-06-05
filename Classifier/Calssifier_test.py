import numpy as np
import mne
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data()

# Use the same preprocessing pipeline from your script
raw_intensity.annotations.set_durations(5)
raw_intensity.annotations.rename({"1.0": "Control", "2.0": "Tapping/Left", "3.0": "Tapping/Right"})
unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
raw_intensity.annotations.delete(unwanted)

picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(raw_intensity.info, picks=picks)
raw_intensity.pick(picks[dists > 0.01])

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
raw_od.info["bads"] = list(np.array(raw_od.ch_names)[sci < 0.5])
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

events, event_dict = mne.events_from_annotations(raw_haemo)
reject_criteria = dict(hbo=80e-6)
tmin, tmax = -5, 15
epochs = mne.Epochs(
    raw_haemo, events, event_id=event_dict, tmin=tmin, tmax=tmax,
    reject=reject_criteria, reject_by_annotation=True, proj=True, 
    baseline=(None, 0), preload=True, detrend=None, verbose=True
)

# **Feature Extraction (No File Modifications)**
hbo_data = epochs.get_data(picks="hbo")  # Shape (n_epochs, n_channels, timepoints)
hbr_data = epochs.get_data(picks="hbr")

# Reshape (Flatten each epoch into a feature vector)
X_hbo = hbo_data.reshape(hbo_data.shape[0], -1)  
X_hbr = hbr_data.reshape(hbr_data.shape[0], -1)
X = np.hstack((X_hbo, X_hbr))  # Combine HbO & HbR

# Extract event labels, keeping only valid epochs
event_ids = epochs.events[:, -1]  # Get event codes for retained epochs
y = np.array([1 if event_dict["Tapping/Left"] == e or event_dict["Tapping/Right"] == e else 0 for e in event_ids])

# **Apply PCA for Feature Reduction**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=20)  # Reduce dimensionality
X_pca = pca.fit_transform(X_scaled)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# **Neural Network Model**
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # Binary classification
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save Model & Transformers
model.save("fnirs_tapping_nn.h5")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")