import os
import numpy as np
from numpy.random import default_rng
from scipy.stats import ttest_rel
from tqdm import tqdm
from itertools import compress
import mne
from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
    scalp_coupling_index,
    temporal_derivative_distribution_repair
)

# --- I/O & EPOCHING --------------------------------------------------------

def read_snirf(snirf_path):
    """
    Read a SNIRF file, convert to HbO/HbR, return an MNE Raw object.
    """
    print(f"Loading {snirf_path}")
    raw = mne.io.read_raw_snirf(snirf_path, preload=True)
    raw_od = optical_density(raw)
    # Check the quality of the coupling between the scalp and the optodes
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    # Convert from optical density to haemoglobin concentration using the Beer-Lambert law
    raw_haemo = beer_lambert_law(raw_od)
    # Filtering: apply band pass filter to remove heartbeat and slow drifts
    raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02, verbose=False)

    return raw_haemo


def make_epochs(raw_hb, tmin=-5.0, tmax=15.0, baseline=(None, 0)):
    """
    Epoch raw_hb around two annotation types:
      '1'          → activation
      'gap_period' → control
    Returns a dict: {'activation': Epochs, 'control': Epochs}
    """
    tmin = -5.0
    tmax = 15.0

    ann_map = {'1': 1, 'control': 2}
    events, used_id = mne.events_from_annotations(raw_hb, event_id=ann_map)

    # Define rejection criteria: any channel exceeding 80e-6 is considered too noisy.
    reject_criteria = dict(hbo=80e-6)
    epochs = mne.Epochs(
        raw_haemo,
        events,
        event_id=used_id,
        tmin=tmin,
        tmax=tmax,
        reject=reject_criteria,
        reject_by_annotation=True,
        proj=True,
        baseline=(None, 0),
        # baseline=(0, 0), 
        preload=True,
        detrend=None,
        verbose=True,
    )

    return {
        'activation': epochs['1'],
        'control':    epochs['control']
    }


# # --- STATISTICAL TESTS -----------------------------------------------------
def extract_epoch_means(epochs_obj, time_window, pick_type):
    """
    Compute the mean power for each epoch over channels and a time window.
    
    Parameters:
        epochs_obj: MNE Epochs object.
        time_window: tuple (t_min, t_max) in seconds.
        pick_type: string (e.g., 'hbo' or 'hbr').
    
    Returns:
        1D array with mean power (i.e., average of the squared signal) for each epoch.
    """
    data = epochs_obj.get_data(picks=pick_type)
    idx = np.where((epochs_obj.times >= time_window[0]) & (epochs_obj.times <= time_window[1]))[0]
    return np.mean(np.square(data[:, :, idx]), axis=(1, 2))



# --- Helper functions (existing) ---
def _rms(x):
    """Root-mean-square of an array."""
    return np.sqrt(np.mean(np.square(x)))

def avg_rms_pick(epochs, pick, time_window):
    """
    RMS of the channel-averaged signal using a specific pick.
    """
    evoked = epochs.average(picks=pick)
    idx = np.where((evoked.times >= time_window[0]) & (evoked.times <= time_window[1]))[0]
    return _rms(evoked.data[:, idx])

def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
    """
    Non-parametric permutation test on the RMS of the channel-averaged signal for a given pick.
    Returns the observed difference (A − B) and a two-sided p-value.
    """
    rng = default_rng(seed)
    all_ep = mne.concatenate_epochs([epochs_a, epochs_b])
    n_a = len(epochs_a)
    observed = avg_rms_pick(epochs_a, pick, time_window) - avg_rms_pick(epochs_b, pick, time_window)
    null_dist = np.empty(n_perm)
    for i in tqdm(range(n_perm), desc=f"Permutations ({pick})"):
        idx = rng.permutation(len(all_ep))
        ep_a = all_ep[idx[:n_a]]
        ep_b = all_ep[idx[n_a:]]
        null_dist[i] = avg_rms_pick(ep_a, pick, time_window) - avg_rms_pick(ep_b, pick, time_window)
    p_val = (np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p_val

def _split_random_half(epochs, *, seed=99):
    """Randomly split an Epochs object into two equal halves (±1 epoch)."""
    rng = default_rng(seed)
    idx = rng.permutation(len(epochs))
    half = len(epochs) // 2
    return epochs[idx[:half]], epochs[idx[half:half*2]]

# --- New helper for average power test (modified) ---
def average_power_test(epochs_a, epochs_b, time_window, pick):
    """
    Compute a paired t-test for the average power from each epoch.
    The average power is computed over the specified time window using the
    existing extract_epoch_means function.
    If the groups have unequal number of epochs, truncate to the smallest sample.
    Returns the paired t-test p-value.
    """
    # Use the already-defined extract_epoch_means from the notebook
    data_a = extract_epoch_means(epochs_a, time_window, pick)
    data_b = extract_epoch_means(epochs_b, time_window, pick)
    
    # Equalise trial counts
    n = min(len(data_a), len(data_b))
    data_a, data_b = data_a[:n], data_b[:n]
    
    _, p_val = ttest_rel(data_a, data_b)
    return p_val

def tipplets_combined_p(p1, p2):
    """
    Combine two p-values using Tippett's method:
        p_combined = 1 - (1 - min(p1, p2)) ** 2
    """
    return 1 - (1 - min(p1, p2))**2

# --- Combined test function ---
def combined_test(epochs_a, epochs_b, time_window, pick, n_perm=5000, seed=42):
    """
    Runs both the phase-alignment permutation test and the average power test.
    Returns a dictionary with p-values:
        'p_phase': phase test p-value,
        'p_power': average power test p-value,
        'p_combined': Tippett's combined p-value.
    """
    _, p_phase = permutation_rms_test_pick(epochs_a, epochs_b, pick=pick, time_window=time_window, n_perm=n_perm, seed=seed)
    p_power = average_power_test(epochs_a, epochs_b, time_window, pick)
    p_comb = tipplets_combined_p(p_phase, p_power)
    return {'p_phase': p_phase, 'p_power': p_power, 'p_combined': p_comb}

def replace_fraction_with_control(tap_epochs, control_epochs, frac, *, seed=123):
    """
    Return a new Epochs object in which a fraction (`frac`) of `tap_epochs`
    has been replaced by randomly-selected epochs from `control_epochs`.
    The total number of epochs and their order are preserved.
    """
    assert 0.0 <= frac <= 1.0, "Fraction must be between 0 and 1."
    rng = default_rng(seed)

    n_total = len(tap_epochs)
    n_replace = int(round(frac * n_total))
    if n_replace == 0:
        return tap_epochs.copy()  # nothing to replace

    n_keep = n_total - n_replace
    tap_keep_idx = rng.choice(n_total, size=n_keep, replace=False)
    ctrl_idx = rng.choice(len(control_epochs), size=n_replace, replace=False)

    mixed = mne.concatenate_epochs([tap_epochs[tap_keep_idx],
                                    control_epochs[ctrl_idx]])
    mixed = mixed[rng.permutation(len(mixed))]  # shuffle order

    return mixed

def extract_epoch_means_np(data, times, time_window):
    """
    Compute the mean power (mean of squared signal) for each epoch 
    over the given time window.
    Parameters:
        data      : np.array of shape (n_epochs, n_channels, n_times)
        times     : 1D array of time points
        time_window: tuple (t_min, t_max)
    Returns:
        1D np.array of mean power values for each epoch.
    """
    idx = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]
    return np.mean(np.square(data[:, :, idx]), axis=(1, 2))


def compute_avg_rms_np(data, times, time_window):
    """
    Compute the RMS of the channel–averaged signal.
    data: np.array of shape (n_epochs, n_channels, n_times)
    times: 1D np.array
    time_window: tuple (t_min, t_max)
    """
    evoked = np.mean(data, axis=0)  # shape: (n_channels, n_times)
    idx = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]
    return np.sqrt(np.mean(evoked[:, idx] ** 2))


def permutation_rms_test_pick_np(data_a, data_b, times, time_window, n_perm=5000, seed=42):
    """
    Non-parametric permutation test on the RMS of the channel-averaged signal.
    Parameters:
       data_a, data_b: np.array of shape (n_epochs, n_channels, n_times)
       times         : 1D np.array of time points (assumed common for both)
       time_window   : tuple (t_min, t_max)
    Returns:
       observed difference and two-sided p-value.
    """
    rng = default_rng(seed)
    min_t = min(data_a.shape[2], data_b.shape[2])
    data_a = data_a[:, :, :min_t]
    data_b = data_b[:, :, :min_t]
    # Use cropped time vector
    times_crop = times[:min_t]
    n_a = data_a.shape[0]
    observed = compute_avg_rms_np(data_a, times_crop, time_window) - compute_avg_rms_np(data_b, times_crop, time_window)
    all_data = np.concatenate([data_a, data_b], axis=0)
    n_total = all_data.shape[0]
    null_dist = np.empty(n_perm)
    for i in tqdm(range(n_perm), desc="Permutations", leave=False):
        perm_idx = rng.permutation(n_total)
        sel_a = all_data[perm_idx[:n_a]]
        sel_b = all_data[perm_idx[n_a:]]
        diff = compute_avg_rms_np(sel_a, times_crop, time_window) - compute_avg_rms_np(sel_b, times_crop, time_window)
        null_dist[i] = diff
    p_val = (np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p_val


def average_power_test_np(data_a, data_b, times, time_window):
    """
    Compute a paired t-test for the average power from each epoch.
    Operates on NumPy arrays.
    """
    means_a = extract_epoch_means_np(data_a, times, time_window)
    means_b = extract_epoch_means_np(data_b, times, time_window)
    n = min(len(means_a), len(means_b))
    _, p_val = ttest_rel(means_a[:n], means_b[:n])
    return p_val


def tipplets_combined_p(p1, p2):
    """
    Combine two p–values using Tippett's method.
    """
    return 1 - (1 - min(p1, p2))**2


def combined_test_np(data_a, data_b, times, time_window, n_perm=5000, seed=42):
    """
    Runs both the phase-alignment permutation test and the average power test,
    operating on NumPy arrays.
    Returns a dictionary with p-values.
    """
    _, p_phase = permutation_rms_test_pick_np(data_a, data_b, times, time_window, n_perm=n_perm, seed=seed)
    p_power = average_power_test_np(data_a, data_b, times, time_window)
    p_comb = tipplets_combined_p(p_phase, p_power)
    return {'p_phase': p_phase, 'p_power': p_power, 'p_combined': p_comb}


def pick_channels_np(info, pick):
    """
    Return a list of channel indices whose name contains the substring given by pick.
    For example, passing "hbo" or "hbr" filters channels accordingly.
    """
    return [i for i, ch in enumerate(info['ch_names']) if pick.lower() in ch.lower()]


# --- MAIN ---
script_dir = os.path.dirname(__file__)
snirf_path = os.path.abspath(
    os.path.join(script_dir, '..', 'Data', '3_tongue.snirf')
)
raw_haemo = read_snirf(snirf_path)
time_window = (0, 13)
epochs_dict = make_epochs(raw_haemo,
                          tmin=time_window[0],
                          tmax=time_window[1])
epochs = epochs_dict
CONTROL_REPLACEMENT_FRAC = 0.00
PERMUTATIONS = 5_000

# color_dict = {
# "Tapping_Left/HbO": "#AA3377",
# "Tapping_Left/HbR": "b",
# "Tapping_Right/HbO": "#EE7733",
# "Tapping_Right/HbR": "g",
# "Control/HbO": "#AA3377",
# "Control/HbR": "b",
# }
# styles_dict = dict(Control=dict(linestyle="dashed"))

# # Plot evoked comparisons using MNE's viz
# mne.viz.plot_compare_evokeds(
#     evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict
# )

# --- Create “mixed” activation dataset ------------------------------------
act_mixed = replace_fraction_with_control(
    epochs["activation"],
    epochs["control"],
    CONTROL_REPLACEMENT_FRAC,
    seed=42
)
# After creating epochs and the mixed activation dataset, convert to np arrays.
act_data = act_mixed.get_data()      # shape: (n_epochs, n_channels, n_times)
ctrl_data = epochs["control"].get_data()
times = act_mixed.times  # assuming same time vector for all epochs

# Select channels by type (e.g., "hbo" and "hbr") using the channel names stored in info.
hbo_idx = pick_channels_np(act_mixed.info, "hbo")
hbr_idx = pick_channels_np(act_mixed.info, "hbr")

# For HbO:
act_data_hbo = act_data[:, hbo_idx, :]
ctrl_data_hbo = ctrl_data[:, hbo_idx, :]
res_hbo = combined_test_np(act_data_hbo, ctrl_data_hbo, times, time_window, n_perm=PERMUTATIONS, seed=42)
print(f"Mixed Activation vs. Control (HbO): combined p = {res_hbo['p_combined']:.4g} "
      f"(phase p = {res_hbo['p_phase']:.4g}, power p = {res_hbo['p_power']:.4g})")

# For HbR:
act_data_hbr = act_data[:, hbr_idx, :]
ctrl_data_hbr = ctrl_data[:, hbr_idx, :]
res_hbr = combined_test_np(act_data_hbr, ctrl_data_hbr, times, time_window, n_perm=PERMUTATIONS, seed=42)
print(f"Mixed Activation vs. Control (HbR): combined p = {res_hbr['p_combined']:.4g} "
      f"(phase p = {res_hbr['p_phase']:.4g}, power p = {res_hbr['p_power']:.4g})")

# Final p-value summary:
combined_p_values = [res_hbo['p_combined'], res_hbr['p_combined']]
min_combined = min(combined_p_values)
print("\nSmallest combined p-value:", min_combined)

if min_combined > 0.05:
    all_p_values = [
        res_hbo['p_phase'], res_hbo['p_power'], res_hbo['p_combined'],
        res_hbr['p_phase'], res_hbr['p_power'], res_hbr['p_combined']
    ]
    min_all = min(all_p_values)
    print("Since smallest combined p-value > 0.05, smallest overall p-value:", min_all)

# --- Sanity-check: Control split tests using np arrays ---------------
# Convert control epochs to np arrays:
ctrl_data_split = ctrl_data  # reuse previously obtained data
# For HbO:
# Split the control data into two halves (using permutation on the first axis)
rng = default_rng(99)
n_half = len(ctrl_data_split) // 2
shuffled_idx = rng.permutation(len(ctrl_data_split))
c_half1 = ctrl_data_split[shuffled_idx[:n_half]]
c_half2 = ctrl_data_split[shuffled_idx[n_half: n_half*2]]
ctrl_hbo1 = c_half1[:, hbo_idx, :]
ctrl_hbo2 = c_half2[:, hbo_idx, :]
result_ctrl_hbo = combined_test_np(ctrl_hbo1, ctrl_hbo2, times, time_window, n_perm=PERMUTATIONS, seed=42)
print(f"Control Split (HbO): combined p = {result_ctrl_hbo['p_combined']:.4g} "
      f"(phase p = {result_ctrl_hbo['p_phase']:.4g}, power p = {result_ctrl_hbo['p_power']:.4g})")

# For HbR:
ctrl_hbr1 = c_half1[:, hbr_idx, :]
ctrl_hbr2 = c_half2[:, hbr_idx, :]
result_ctrl_hbr = combined_test_np(ctrl_hbr1, ctrl_hbr2, times, time_window, n_perm=PERMUTATIONS, seed=42)
print(f"Control Split (HbR): combined p = {result_ctrl_hbr['p_combined']:.4g} "
      f"(phase p = {result_ctrl_hbr['p_phase']:.4g}, power p = {result_ctrl_hbr['p_power']:.4g})")
