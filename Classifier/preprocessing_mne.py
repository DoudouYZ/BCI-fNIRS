from itertools import compress

import matplotlib.pyplot as plt
import numpy as np

import mne

#Downloading and loading dataset
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

# Removin all "short channels" that are too close together to detect neural responses. Then visualizing the data
picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks
)
raw_intensity.pick(picks[dists > 0.01])

# Converting raw intensity to optical density. Raw intensity is the raw data from the fNIRS device
# nd optical density is the log of the ratio of light intensities between the detectors and sources.
# This is done to remove the effects of light scattering and absorption in the scalp and skull.
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

# Checking the quality of the coupling between the scalp and the optodes. Here is it clean data so we don't remove bad channels
sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
# (All channels with SCI < 0.5 are considered bad)
raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))

# Converting from optical density to haemoglobin concentration. This is done using the Beer-Lambert Law
# Done to get quantitative measure of relative changes in blood oxygenation
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

# Applying band pass filter to remove heartrate and slow drifts
raw_haemo_unfiltered = raw_haemo.copy()
raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

# Defining range of epochs (events from above), rejection criteria, baseline correction, and extracting epochs
events, event_dict = mne.events_from_annotations(raw_haemo)
reject_criteria = dict(hbo=80e-6) # any signal exceeding 80e-6 is considered too noisy
tmin, tmax = -5, 15
epochs = mne.Epochs(
    raw_haemo,
    events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    reject=reject_criteria,
    reject_by_annotation=True,
    proj=True,
    baseline=(None, 0), # normalizing based on the average of the start of the epoch until time 0
    preload=True,
    detrend=None,
    verbose=True,
)