import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import mne
import mne
from numpy.random import default_rng
import numpy as np
from scipy.stats import ttest_rel
from tqdm import tqdm 
# Insert parent folder for custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preprocessing.preprocessing_mne import get_raw_subject_data

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def load_data(subject=3, time_window=(-5, 15)):
    """
    Load data, perform averaging tests and plots for evoked responses,
    and return concatenated data along with original epoch information.
    
    This function retains all mean tests and plots of averaged epochs.
    ICA-related functionality has been removed.
    
    Returns:
        tuple: (X, orig_indices, y, control_data, left_data, right_data, numb_samples)
    """

    # tmin = time_window[0]
    # tmax = time_window[1]
    tmin = -5.0
    tmax = 15.0

    # Load subject epochs
    epochs = get_raw_subject_data(subject=subject, tmin=tmin, tmax=tmax)

    # Compute evoked responses for each condition and rename channels
    evoked_dict = {
        "Tapping_Left/HbO": epochs["Tapping_Left"].average(picks="hbo"),
        "Tapping_Left/HbR": epochs["Tapping_Left"].average(picks="hbr"),
        "Tapping_Right/HbO": epochs["Tapping_Right"].average(picks="hbo"),
        "Tapping_Right/HbR": epochs["Tapping_Right"].average(picks="hbr"),
        "Control/HbO": epochs["Control"].average(picks="hbo"),
        "Control/HbR": epochs["Control"].average(picks="hbr"),
    }
    for condition in evoked_dict:
        evoked_dict[condition].rename_channels(lambda x: x[:-4])

    return epochs, evoked_dict


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

def avg_rms_pick_window(epochs, pick, time_window):
    """
    RMS of the channel-averaged signal computed only within a time window.
    """
    evoked = epochs.average(picks=pick)
    idx = np.where((evoked.times >= time_window[0]) & (evoked.times <= time_window[1]))[0]
    return _rms(evoked.data[:, idx])

def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
    """
    Non-parametric permutation test on the RMS of the channel-averaged signal for a given pick,
    computed only over the specified time window.
    Returns the observed difference (A − B) and a two-sided p-value.
    """
    rng = default_rng(seed)
    all_ep = mne.concatenate_epochs([epochs_a, epochs_b])
    n_a = len(epochs_a)
    observed = avg_rms_pick_window(epochs_a, pick, time_window) - avg_rms_pick_window(epochs_b, pick, time_window)
    null_dist = np.empty(n_perm)
    for i in tqdm(range(n_perm), desc=f"Permutations ({pick})"):
        idx = rng.permutation(len(all_ep))
        ep_a = all_ep[idx[:n_a]]
        ep_b = all_ep[idx[n_a:]]
        null_dist[i] = avg_rms_pick_window(ep_a, pick, time_window) - avg_rms_pick_window(ep_b, pick, time_window)
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

# --- MAIN ---
subject = 0
time_window = (0, 13)
epochs, evoked_dict = load_data(subject=subject, time_window=time_window)
CONTROL_REPLACEMENT_FRAC = 0.70
PERMUTATIONS = 5000

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

# --- Create “mixed” tapping datasets -----------------------------------------
tap_left_mixed  = replace_fraction_with_control(epochs["Tapping_Left"],  epochs["Control"],
                                                CONTROL_REPLACEMENT_FRAC, seed=42)
tap_right_mixed = replace_fraction_with_control(epochs["Tapping_Right"], epochs["Control"],
                                                CONTROL_REPLACEMENT_FRAC, seed=42)

# --- Run combined tests with the mixed tapping data --------------------------
print(f"\n--- Mixed Tapping (fraction replaced: {CONTROL_REPLACEMENT_FRAC:.2f}) vs. Control ---")

# Tapping_Left (mixed)
res_left_hbo_mix = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbo", n_perm=PERMUTATIONS)
print(f"Mixed Tapping_Left vs. Control (HbO): combined p = {res_left_hbo_mix['p_combined']:.4g} "
      f"(phase p = {res_left_hbo_mix['p_phase']:.4g}, power p = {res_left_hbo_mix['p_power']:.4g})")

res_left_hbr_mix = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbr", n_perm=PERMUTATIONS)
print(f"Mixed Tapping_Left vs. Control (HbR): combined p = {res_left_hbr_mix['p_combined']:.4g} "
      f"(phase p = {res_left_hbr_mix['p_phase']:.4g}, power p = {res_left_hbr_mix['p_power']:.4g})")

# Tapping_Right (mixed)
res_right_hbo_mix = combined_test(tap_right_mixed, epochs["Control"], time_window, pick="hbo", n_perm=PERMUTATIONS)
print(f"Mixed Tapping_Right vs. Control (HbO): combined p = {res_right_hbo_mix['p_combined']:.4g} "
      f"(phase p = {res_right_hbo_mix['p_phase']:.4g}, power p = {res_right_hbo_mix['p_power']:.4g})")

res_right_hbr_mix = combined_test(tap_right_mixed, epochs["Control"], time_window, pick="hbr", n_perm=PERMUTATIONS)
print(f"Mixed Tapping_Right vs. Control (HbR): combined p = {res_right_hbr_mix['p_combined']:.4g} "
      f"(phase p = {res_right_hbr_mix['p_phase']:.4g}, power p = {res_right_hbr_mix['p_power']:.4g})")

# Final p-value
# Collect combined p-values from the mixed tests
combined_p_values = [
    res_left_hbo_mix['p_combined'], res_left_hbr_mix['p_combined'],
    res_right_hbo_mix['p_combined'], res_right_hbr_mix['p_combined']
]
min_combined = min(combined_p_values)
print("\nSmallest combined p-value:", min_combined)

# If the smallest combined p-value is greater than 0.05, print the smallest of all p-values
if min_combined > 0.05:
    all_p_values = [
        res_left_hbo_mix['p_phase'], res_left_hbo_mix['p_power'], res_left_hbo_mix['p_combined'],
        res_left_hbr_mix['p_phase'], res_left_hbr_mix['p_power'], res_left_hbr_mix['p_combined'],
        res_right_hbo_mix['p_phase'], res_right_hbo_mix['p_power'], res_right_hbo_mix['p_combined'],
        res_right_hbr_mix['p_phase'], res_right_hbr_mix['p_power'], res_right_hbr_mix['p_combined']
    ]
    min_all = min(all_p_values)
    print("Since smallest combined p-value > 0.05, smallest overall p-value:", min_all)


# --- Sanity-check: Control split tests. Split the Control epochs and run the combined test.
print("\n--- Control Split Tests ---")
control_epochs = epochs["Control"]

# For HbO
c_half1, c_half2 = _split_random_half(control_epochs, seed=99)
result_ctrl_hbo = combined_test(c_half1, c_half2, time_window, pick="hbo", n_perm=PERMUTATIONS)
print(f"Control Split (HbO): combined p = {result_ctrl_hbo['p_combined']:.4g} "
      f"(phase p = {result_ctrl_hbo['p_phase']:.4g}, power p = {result_ctrl_hbo['p_power']:.4g})")

# For HbR
c_half1, c_half2 = _split_random_half(control_epochs, seed=99)
result_ctrl_hbr = combined_test(c_half1, c_half2, time_window, pick="hbr", n_perm=PERMUTATIONS)
print(f"Control Split (HbR): combined p = {result_ctrl_hbr['p_combined']:.4g} "
      f"(phase p = {result_ctrl_hbr['p_phase']:.4g}, power p = {result_ctrl_hbr['p_power']:.4g})")
