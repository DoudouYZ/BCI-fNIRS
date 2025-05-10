from itertools import compress
import numpy as np
import mne
import mne_nirs
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


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

    # Convert raw intensity to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od, 0.01)

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

def get_epochs_for_subject(subject: int = 0, add_hbr=False, hbr_multiplier=1.0, hbr_shift=0.0, tmin=-5, tmax=15):
    """
    Generate MNE epochs for a given subject by loading, preprocessing, and based on the options, augmenting the data with shifted HbR information.
    This function performs the following steps:
    1. Loads raw intensity data for the specified subject.
    2. Preprocesses the raw data to obtain haemodynamic signals.
    3. Extracts epochs from the preprocessed data within the time window [tmin, tmax].
    4. Optionally, if add_hbr is True:
        - Multiplies the HbR component in the epochs by the given multiplier.
        - Identifies channel indices for both HbO and HbR.
        - Shifts the HbR data by an amount corresponding to hbr_shift seconds.
        - Adds the shifted HbR data to the HbO channels.
    Parameters:
         subject (int, optional): Subject identifier for which the raw data is to be loaded. Defaults to 0.
         add_hbr (bool, optional): Flag indicating whether to modify the epochs by adding shifted HbR data to the HbO channels. Defaults to False.
         hbr_multiplier (float, optional): A multiplier applied to the HbR data when enhancing the epochs. Defaults to 1.0.
         hbr_shift (float, optional): Time shift in seconds applied to the HbR data. The shift is converted into samples based on the sampling frequency. Defaults to 0.0.
         tmin (float or int, optional): Start time of the epoch relative to the event onset. Defaults to -5.
         tmax (float or int, optional): End time of the epoch relative to the event onset. Defaults to 15.
    Returns:
         mne.Epochs: The processed epochs object containing the extracted epochs with optional HbR augmentation.
    """


    raw_intensity = load_raw_data(subject)
    raw_haemo = preprocess_raw_data(raw_intensity)
    epochs = extract_epochs(raw_haemo, tmin, tmax)

    if add_hbr:
        epochs = multiply_hbr_in_epochs(epochs, hbr_multiplier, 3*10**(-6)) # 3*10**(-6) looks to be a good boundary fo HbR
        # Identify channel indices for HbO and HbR
        hbo_idx = [i for i, ch in enumerate(epochs.ch_names) if 'hbo' in ch.lower()]
        hbr_idx = [i for i, ch in enumerate(epochs.ch_names) if 'hbr' in ch.lower()]

        if hbr_idx:
            sfreq = epochs.info['sfreq']
            sample_shift = int(round(hbr_shift * sfreq))
            # Shift the HbR data to the right by 2 seconds
            hbr_data = epochs._data[:, hbr_idx, :]
            shifted_hbr = np.zeros_like(hbr_data)
            if sample_shift < hbr_data.shape[2]:
                shifted_hbr[:, :, sample_shift:] = hbr_data[:, :, :-sample_shift]
            else:
                shifted_hbr = hbr_data  # fallback if shift is too large
            # Add the shifted HbR data to the HbO channels
            if hbo_idx:
                epochs._data[:, hbo_idx, :] += shifted_hbr

    return epochs

def get_group_epochs_subtracting_short(num_subjects: int = 5, add_hbr=False, hbr_multiplier=1.0, hbr_shift=0.0, tmin=-5, tmax=15):
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
        epochs.append(get_epochs_for_subject(subject, add_hbr=add_hbr, hbr_multiplier=hbr_multiplier, hbr_shift=hbr_shift, tmin=tmin, tmax=tmax))
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


def multiply_hbr_in_epochs(epochs, factor, boundary):
    """
    Multiplies the HbR channel data by the given scalar factor within the epochs,
    but only for values less than -2.
    
    Args:
        epochs: The epochs object containing haemoglobin data.
        factor: Scalar factor to multiply the HbR signal.
        
    Returns:
        The modified epochs object.
    """
    # Identify channel indices that contain "hbr" (case insensitive)
    hbr_idx = [i for i, ch in enumerate(epochs.ch_names) if 'hbr' in ch.lower()]
    if hbr_idx:
        # Multiply only the HbR signal entries < -2 by the given factor
        # print("HbR sample boolean:", epochs._data[0, hbr_idx, :10] < boundary)
        data = epochs._data[:, hbr_idx, :]
        mask = data < boundary
        data[mask] *= factor
        epochs._data[:, hbr_idx, :] = data
    return epochs


if __name__ == '__main__':
    # 0.128 for full sample rate
    epochs = get_epochs_for_subject(0, add_hbr=True, hbr_multiplier=5.0, hbr_shift=4.0, tmin=-10, tmax=15)

    epochs['Tapping_Right'].plot_image(
        combine="mean",
        vmin=-30,
        vmax=30,
        ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
    )