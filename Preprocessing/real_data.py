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
data_file = os.path.join(base_dir, "..", "Data", "2_hand.snirf")

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
    # Don't modify the original data directly
    raw_copy = raw_intensity.copy()

    # Pick and keep only channels with valid source-detector distances
    picks = mne.pick_types(raw_copy.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(raw_copy.info, picks=picks)
    raw_copy.pick(picks[dists > 0.01])
    
    # Convert raw intensity to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw_copy)

    # Compute scalp coupling index and drop bad channels
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    good_channels = list(compress(raw_od.ch_names, sci >= 0.5))
    raw_od.pick(good_channels)

    # Convert to haemoglobin concentration using Beer-Lambert law
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)

    # Apply band-pass filter
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

# Plot time series of raw intensity data

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)
plt.show()


# if __name__ == "__main__":
#     # Preprocess
#     raw_haemo = preprocess_raw_data(raw_intensity)
    
#     # Extract epochs
#     epochs = extract_epochs(raw_haemo)

#     # Plot evoked response
#     times = np.arange(-3.5, 13.2, 3.0)
#     topomap_args = dict(extrapolate="local")

#     # Check available event keys
#     print("Available event types:", epochs.event_id)

#     # Plot for "1" condition
#     if "control" in epochs.event_id:
#         epochs["control"].average(picks="hbo").plot_joint(
#             times=times, topomap_args=topomap_args
#         )
#         plt.show()
#     else:
#         print("⚠️ 'Tapping' event not found in epochs.")