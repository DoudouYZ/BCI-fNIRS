from itertools import compress
import numpy as np
import mne
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

# Yuxuan : 1
# Oscar : 2
# Nikolaj : 3


# Get path to this script
base_dir = os.path.dirname(__file__)

# Build relative path to data file
data_file = os.path.join(base_dir, "..", "Data", "1_tongue.snirf")

# Load SNIRF data
raw_intensity = mne.io.read_raw_snirf(data_file, preload=True)

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
    raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02, verbose=False)

    return raw_haemo

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
    raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02, verbose=False)

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
    events, event_dict = mne.events_from_annotations(raw_haemo, verbose=False)
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
        verbose=False,
    )
    return epochs

def get_raw_subject_data(subject: int = 0, tmin=-5, tmax=15, force_download=False):
    raw_intensity = load_raw_data(subject, force_download=force_download)
    raw_haemo = preprocess_raw_data(raw_intensity)
    epochs = extract_epochs(raw_haemo, tmin, tmax)
    return epochs