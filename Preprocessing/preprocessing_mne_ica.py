import numpy as np
import mne
from mne.preprocessing import ICA
from mne.datasets import fnirs_motor
import matplotlib.pyplot as plt

def load_raw_data():
    """
    Downloads and loads the fNIRS dataset and performs initial annotations.
    Returns:
        raw_intensity: The raw intensity MNE object.
    """
    # Downloading and loading dataset
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
    raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
    raw_intensity.load_data()

    # Annotating and removing unnecessary trigger codes
    raw_intensity.annotations.set_durations(5)
    raw_intensity.annotations.rename(
        {"1.0": "Control", "2.0": "Tapping/Left", "3.0": "Tapping/Right"}
    )
    unwanted = np.nonzero(raw_intensity.annotations.description == "15.0")
    raw_intensity.annotations.delete(unwanted)

    return raw_intensity


def apply_ica_to_condition(raw, condition_labels):
    """
    Applies ICA to specific conditions from the dataset.
    
    Args:
        raw: The loaded fNIRS dataset.
        condition_labels (list): List of condition names to extract and process (e.g., ["Tapping/Left", "Tapping/Right"]).
    
    Returns:
        ica: The fitted ICA object.
        raw_clean: The ICA-cleaned MNE object.
    """
    from mne.preprocessing import ICA
    
    # Convert to Optical Density (OD)
    raw_od = mne.preprocessing.nirs.optical_density(raw)

    # Apply bandpass filtering
    raw_od.filter(l_freq=0.01, h_freq=0.2, fir_design="firwin")

    # Convert to hemoglobin concentration
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)

    # Extract the selected condition epochs
    events, event_dict = mne.events_from_annotations(raw_haemo)
    
    # Ensure selected conditions exist in the dataset
    missing_conditions = [c for c in condition_labels if c not in event_dict]
    if missing_conditions:
        raise ValueError(f"Conditions {missing_conditions} not found. Available conditions: {list(event_dict.keys())}")

    # Select multiple conditions
    selected_event_ids = {c: event_dict[c] for c in condition_labels}
    condition_epochs = mne.Epochs(raw_haemo, events, event_id=selected_event_ids, 
                                  tmin=0, tmax=5, baseline=None, preload=True)

    # Apply ICA to the combined conditions
    ica = ICA(n_components=10, method="fastica", random_state=42)
    ica.fit(condition_epochs)  # ✅ ICA now works on multiple conditions

    # Apply ICA to clean the data
    raw_clean = raw_haemo.copy()
    ica.apply(raw_clean)

    return ica, raw_clean


# Choose a condition: "Control", "Tapping/Left", or "Tapping/Right"
condition_to_use = ["Tapping/Right", "Tapping/Left"] # Change as needed

# Load the dataset
raw_intensity = load_raw_data()

# Apply ICA to the selected condition
ica, raw_clean = apply_ica_to_condition(raw_intensity, condition_to_use)

# **Plot ICA components for the selected condition**
ica.plot_components()  # This will generate a plot like your uploaded ICA image