import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import mne
import mne
from numpy.random import default_rng
import numpy as np
from scipy.stats import ttest_rel
from tqdm import tqdm 
# Insert parent folder for custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
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
def compute_avg_rms(data, times, time_window):
    # Compute the average signal (evoked) across epochs
    evoked = np.mean(data, axis=0)  # shape: (n_channels, n_times)
    # Identify indices within the desired time window
    idx = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]
    # Return RMS over selected time points
    return np.sqrt(np.mean(evoked[:, idx] ** 2))

def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
    """
    Non-parametric permutation test on the RMS of the channel-averaged signal for a given pick.
    This version extracts the underlying data, crops the time axis to the minimum length,
    and then performs the permutation test. Returns the observed difference (A – B) and a two-sided p-value.
    """
    rng = default_rng(seed)
    # Get data arrays for the given pick
    data_a = epochs_a.get_data(picks=pick)  # shape: (n_epochs_a, n_channels, n_times_a)
    data_b = epochs_b.get_data(picks=pick)  # shape: (n_epochs_b, n_channels, n_times_b)
    # Crop time axis to minimum length
    min_t = min(data_a.shape[2], data_b.shape[2])
    data_a = data_a[:, :, :min_t]
    data_b = data_b[:, :, :min_t]
    # Use the time vector from one of the epochs
    times = epochs_a.times[:min_t]

    n_a = data_a.shape[0]
    n_total = data_a.shape[0] + data_b.shape[0]

    observed = compute_avg_rms(data_a, times, time_window) - compute_avg_rms(data_b, times, time_window)

    # Concatenate the two datasets along the epoch dimension
    all_data = np.concatenate([data_a, data_b], axis=0)

    null_dist = np.empty(n_perm)
    for i in tqdm(range(n_perm), desc=f"Permutations ({pick})"):
        perm_idx = rng.permutation(n_total)
        sel_a = all_data[perm_idx[:n_a]]
        sel_b = all_data[perm_idx[n_a:]]
        diff = compute_avg_rms(sel_a, times, time_window) - compute_avg_rms(sel_b, times, time_window)
        null_dist[i] = diff

    p_val = (np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p_val

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
        return tap_epochs.copy()

    n_keep = n_total - n_replace
    tap_keep_idx = rng.choice(n_total, size=n_keep, replace=False)
    ctrl_idx = rng.choice(len(control_epochs), size=n_replace, replace=False)

    mixed = mne.concatenate_epochs([tap_epochs[tap_keep_idx],
                                    control_epochs[ctrl_idx]])
    mixed = mixed[rng.permutation(len(mixed))]  # shuffle order

    return mixed

# --- MAIN ---
time_window = (0, 13)
subjects = [1, 2, 3, 4]
fractions = np.arange(0.0, 0.75 + 0.05, 0.05)

for subj in subjects:
    epochs, evoked_dict = load_data(subject=subj, time_window=time_window)
    results_all = []
    
    for frac in fractions:
        for seed in range(7):
            tap_left_mixed  = replace_fraction_with_control(epochs["Tapping_Left"], epochs["Control"], frac, seed=seed)
            tap_right_mixed = replace_fraction_with_control(epochs["Tapping_Right"], epochs["Control"], frac, seed=seed)
            
            res_left_hbo  = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbo", seed=seed)
            res_left_hbr  = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbr", seed=seed)
            res_right_hbo = combined_test(tap_right_mixed,epochs["Control"], time_window, pick="hbo", seed=seed)
            res_right_hbr = combined_test(tap_right_mixed,epochs["Control"], time_window, pick="hbr", seed=seed)
            
            p_combined_list = [res_left_hbo['p_combined'], res_left_hbr['p_combined'],
                               res_right_hbo['p_combined'], res_right_hbr['p_combined']]
            smallest_combined = min(p_combined_list)
            
            all_p_values = [res_left_hbo['p_phase'], res_left_hbo['p_power'], res_left_hbo['p_combined'],
                            res_left_hbr['p_phase'], res_left_hbr['p_power'], res_left_hbr['p_combined'],
                            res_right_hbo['p_phase'], res_right_hbo['p_power'], res_right_hbo['p_combined'],
                            res_right_hbr['p_phase'], res_right_hbr['p_power'], res_right_hbr['p_combined']]
            smallest_overall = min(all_p_values) if smallest_combined > 0.05 else np.nan
            
            result_row = {
                "subject": subj,
                "CONTROL_REPLACEMENT_FRAC": frac,
                "seed": seed,
                "smallest_combined_p": smallest_combined,
                "smallest_overall_p": smallest_overall,
                "left_hbo_p_phase": res_left_hbo['p_phase'],
                "left_hbo_p_power": res_left_hbo['p_power'],
                "left_hbo_p_combined": res_left_hbo['p_combined'],
                "left_hbr_p_phase": res_left_hbr['p_phase'],
                "left_hbr_p_power": res_left_hbr['p_power'],
                "left_hbr_p_combined": res_left_hbr['p_combined'],
                "right_hbo_p_phase": res_right_hbo['p_phase'],
                "right_hbo_p_power": res_right_hbo['p_power'],
                "right_hbo_p_combined": res_right_hbo['p_combined'],
                "right_hbr_p_phase": res_right_hbr['p_phase'],
                "right_hbr_p_power": res_right_hbr['p_power'],
                "right_hbr_p_combined": res_right_hbr['p_combined']
            }
            results_all.append(result_row)

            print(f"\n Subject {subj}, Fraction {frac:.2f}, Seed {seed} \n")
    
    # Save the file with all individual (frac,seed) results
    df_all = pd.DataFrame(results_all)
    csv_all = f"subject_{subj}_control_replacement_all_results.csv"
    df_all.to_csv(csv_all, index=False)
    
    # Aggregate: compute mean and std for each p-value per fraction
    p_cols = ["smallest_combined_p", "smallest_overall_p",
              "left_hbo_p_phase", "left_hbo_p_power", "left_hbo_p_combined",
              "left_hbr_p_phase", "left_hbr_p_power", "left_hbr_p_combined",
              "right_hbo_p_phase", "right_hbo_p_power", "right_hbo_p_combined",
              "right_hbr_p_phase", "right_hbr_p_power", "right_hbr_p_combined"]
    aggregations = {col: ['mean', 'std'] for col in p_cols}
    df_summary = df_all.groupby("CONTROL_REPLACEMENT_FRAC").agg(aggregations)
    # Flatten the column MultiIndex
    df_summary.columns = ['_'.join(col) for col in df_summary.columns.values]
    df_summary = df_summary.reset_index()
    
    csv_summary = f"subject_{subj}_control_replacement_summary.csv"
    df_summary.to_csv(csv_summary, index=False)