from itertools import compress
import numpy as np
import mne
import mne_nirs
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path

def load_raw_data(subject: int = 0):
    """
    Downloads and loads the fNIRS dataset and performs initial annotations.
    Returns:
        raw_intensity: The raw intensity MNE object.
    """
    # Downloading and loading dataset
    fnirs_data_folder = Path(mne_nirs.datasets.fnirs_motor_group.data_path())

    dataset = BIDSPath(
    root=fnirs_data_folder, task="tapping", datatype="nirs", suffix="nirs", extension=".snirf"
    )
    subjects = get_entity_vals(fnirs_data_folder, "subject")
    bids_path = dataset.update(subject=subjects[subject])
    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    raw_intensity.annotations.delete(raw_intensity.annotations.description == "15.0")
    raw_intensity.annotations.description[:] = [
    d.replace("/", "_") for d in raw_intensity.annotations.description
    ]
    # raw_intensity.load_data()
    
    # Annotating and removing unnecessary trigger codes
    # raw_intensity.annotations.set_durations(5)
    # raw_intensity.annotations.rename(
    #     {"1.0": "Control", "2.0": "Tapping/Left", "3.0": "Tapping/Right"}
    # )

    return raw_intensity


def preprocess_raw_data(raw_intensity):
    """
    Preprocess raw intensity data into haemoglobin concentration
    by applying channel selection, conversion to optical density,
    checking scalp coupling, conversion to haemoglobin, and filtering.
    Args:
        raw_intensity: The raw intensity MNE object.
    Returns:
        raw_haemo: The preprocessed haemoglobin concentration MNE object.
    """
    # Remove short channels for neural responses
    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(raw_intensity.info, picks=picks)
    raw_intensity.pick(picks[dists > 0.01])

    # Convert raw intensity to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    # Check the quality of the coupling between the scalp and the optodes
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))

    # Convert from optical density to haemoglobin concentration using the Beer-Lambert law
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

    # Filtering: apply band pass filter to remove heartbeat and slow drifts
    raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

    return raw_haemo


def extract_epochs(raw_haemo, tmin=-5, tmax=15):
    """
    Extracts epochs from preprocessed data.
    Args:
        raw_haemo: Preprocessed MNE object with haemoglobin concentration.
        tmin: Start time before the event.
        tmax: End time after the event.
    Returns:
        epochs: The extracted epochs.
    """
    events, event_dict = mne.events_from_annotations(raw_haemo)
    # Define rejection criteria: any channel exceeding 80e-6 is considered too noisy.
    reject_criteria = dict(hbo=80e-6)
    epochs = mne.Epochs(
        raw_haemo,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        reject=reject_criteria,
        reject_by_annotation=True,
        proj=True,
        baseline=(None, 0),
        preload=True,
        detrend=None,
        verbose=True,
    )
    return epochs

def get_group_epochs(num_subjects: int = 5, tmin=-5, tmax=15):
    """
    Pipeline for loading, preprocessing, and extracting epochs for a group of subjects.
    Args:
        num_subjects: The number of subjects to process.
        tmin: Start time before the event.
        tmax: End time after the event.
    Returns:
        epochs: The extracted epochs.
    """
    epochs = []
    for subject in range(num_subjects):
        raw_intensity = load_raw_data(subject)
        raw_haemo = preprocess_raw_data(raw_intensity)
        epoch = extract_epochs(raw_haemo, tmin, tmax)
        epochs.append(epoch)
    return epochs

def get_epochs_for_subject(subject: int = 0):
    """
    Pipeline for loading, preprocessing, and extracting epochs for a single subject.
    Args:
        subject: The subject number.
    Returns:
        epochs: The extracted epochs.
    """
    raw_intensity = load_raw_data(subject)
    raw_haemo = preprocess_raw_data(raw_intensity)
    epochs = extract_epochs(raw_haemo)
    return epochs

def compute_segment_mean(data, label_value, seg_samples):
    """
    Computes the mean features for each segment in the provided data.
    Each segment is extracted from an epoch and its mean is calculated
    along the time axis.
    
    Args:
        data (ndarray): Array of epochs with shape (n_epochs, n_channels, n_times)
        label_value (int): Label value to assign to each segment from these epochs

    Returns:
        tuple: A tuple containing:
            - segs (ndarray): Array of mean features for each segment.
            - seg_labels (ndarray): Array of corresponding label values.
    """
    segs, seg_labels = [], []
    # Loop over each epoch in the data
    for epoch in data:
        n_time = epoch.shape[1]  # total time samples in the epoch
        n_segs = n_time // seg_samples  # number of full segments that can be extracted
        # Iterate over each segment in the epoch
        for i in range(n_segs):
            # Extract the segment for all channels
            segment = epoch[:, i * seg_samples : (i + 1) * seg_samples]
            # Compute mean power as the average squared value for each channel in this segment
            mean_val = np.mean(segment, axis=1)
            # Append the computed mean and corresponding label
            segs.append(mean_val)
            seg_labels.append(label_value)
    return np.array(segs), np.array(seg_labels)

# Function to stack the epochs for left tapping and control and split them into s second windows
def stack_epochs(epochs, s, tmin=0, tmax=10):
    """
    Stacks epochs for left tapping and control and splits them into
    s second windows."
    """
    data = epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()
    labels = epochs.events[:, -1]  # getting labels from epoch events
    # Stack epochs for left tapping and control
    left_tapping = data[labels == 2]
    control = data[labels == 1]

    # Get the sampling frequency from the epochs object
    sfreq = epochs.info["sfreq"]
    # Calculate the number of samples per segment based on the window length (s seconds)
    seg_samples = int(s * sfreq)

    # Compute power features for left tapping segments (label 2)
    left_features, left_labels = compute_segment_mean(left_tapping, 2, seg_samples)
    # Compute power features for control segments (label 1)
    control_features, control_labels = compute_segment_mean(control, 1, seg_samples)

    # Concatenate the features and labels from both conditions
    X = np.concatenate([left_features, control_features], axis=0)
    y = np.concatenate([left_labels, control_labels], axis=0)

    return X, y



# Computing the PCA for the stacked epochs
# Compute features and labels from the stacked epochs
window_length = 1  # seconds
X, y = stack_epochs(get_epochs_for_subject(0), s=window_length)

# Perform PCA reducing to 4 components
nr_pcs = 3
pca = PCA(n_components=nr_pcs)
X_pca = pca.fit_transform(X)

# Create a grid of plots for the first 4 principal components
fig, axes = plt.subplots(nr_pcs, nr_pcs, figsize=(12, 12))
pcs = ['PC 1', 'PC 2', 'PC 3']

for i in range(nr_pcs):
    for j in range(nr_pcs):
        ax = axes[i, j]
        if i == j:
            # Plot a histogram on the diagonal
            ax.hist(X_pca[:, i], bins=30, color='gray')
            ax.set_xlabel(pcs[i])
            ax.set_ylabel('Count')
        else:
            # Scatter plot for the off-diagonals
            sc = ax.scatter(X_pca[:, j], X_pca[:, i], c=y, cmap='viridis', alpha=0.7)
            if i == 3:
                ax.set_xlabel(pcs[j])
            if j == 0:
                ax.set_ylabel(pcs[i])

plt.suptitle(f'Pairplot of the First {nr_pcs} Principal Components', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.colorbar(sc, ax=axes.ravel().tolist(), label='Label')
plt.show()