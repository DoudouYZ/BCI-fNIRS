import os
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import mne
from numpy.random import default_rng
from scipy.stats import ttest_rel
from tqdm import tqdm 

# Insert parent folder for custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Preprocessing import get_raw_subject_data

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def load_data(subject=3, time_window=(-5, 15)):
    """
    Load data, perform averaging tests and plots for evoked responses,
    and return concatenated data along with original epoch information.
    """
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
    """
    data = epochs_obj.get_data(picks=pick_type)
    idx = np.where((epochs_obj.times >= time_window[0]) & (epochs_obj.times <= time_window[1]))[0]
    return np.mean(np.square(data[:, :, idx]), axis=(1, 2))

def compute_avg_rms(data, times, time_window):
    evoked = np.mean(data, axis=0)  
    idx = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]
    return np.sqrt(np.mean(evoked[:, idx] ** 2))

def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
    rng = default_rng(seed)
    data_a = epochs_a.get_data(picks=pick)
    data_b = epochs_b.get_data(picks=pick)
    min_t = min(data_a.shape[2], data_b.shape[2])
    data_a = data_a[:, :, :min_t]
    data_b = data_b[:, :, :min_t]
    times = epochs_a.times[:min_t]

    n_a = data_a.shape[0]
    n_total = data_a.shape[0] + data_b.shape[0]

    observed = compute_avg_rms(data_a, times, time_window) - compute_avg_rms(data_b, times, time_window)
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

def average_power_test(epochs_a, epochs_b, time_window, pick):
    data_a = extract_epoch_means(epochs_a, time_window, pick)
    data_b = extract_epoch_means(epochs_b, time_window, pick)
    n = min(len(data_a), len(data_b))
    data_a, data_b = data_a[:n], data_b[:n]
    _, p_val = ttest_rel(data_a, data_b)
    return p_val

def tipplets_combined_p(p1, p2):
    return 1 - (1 - min(p1, p2))**2

def get_channels_by_side(epochs, side):
    """
    Return a list of channel names from an MNE Epochs object
    that belong to a given side ("left" or "right") based on channel location.
    """
    channels = []
    for ch in epochs.info["chs"]:
        loc = ch.get("loc")
        if loc is None or np.all(np.array(loc[:3]) == 0):
            continue
        x = loc[0]
        if side.lower() == "left" and x < 0:
            channels.append(ch["ch_name"])
        elif side.lower() == "right" and x > 0:
            channels.append(ch["ch_name"])
    return channels

def combined_test(epochs_a, epochs_b, time_window, pick, n_perm=5000, seed=42, tapping_side=None, channel_mode="all"):
    """
    Runs both the phase-alignment permutation test and the average power test.
    If tapping_side is provided and channel_mode is "inverted" or "same_side",
    restrict epochs to channels based on their location.
    """
    if tapping_side is not None and channel_mode in ["inverted", "same_side"]:
        side_to_pick = "right" if (channel_mode=="inverted" and tapping_side.lower()=="left") else \
                       "left" if (channel_mode=="inverted" and tapping_side.lower()=="right") else tapping_side.lower()
        channels = get_channels_by_side(epochs_a, side_to_pick)
        epochs_a = epochs_a.copy().pick(channels)
        epochs_b = epochs_b.copy().pick(channels)
    
    _, p_phase = permutation_rms_test_pick(epochs_a, epochs_b, pick=pick, time_window=time_window, n_perm=n_perm, seed=seed)
    p_power = average_power_test(epochs_a, epochs_b, time_window, pick)
    p_comb = tipplets_combined_p(p_phase, p_power)
    return {'p_phase': p_phase, 'p_power': p_power, 'p_combined': p_comb}

def replace_fraction_with_control_no_overlap_both(tap_left_epochs, tap_right_epochs, control_epochs, frac, *, seed=123):
    rng = default_rng(seed)
    n_replace_left = int(round(frac * len(tap_left_epochs)))
    n_replace_right = int(round(frac * len(tap_right_epochs)))
    total_replace = n_replace_left + n_replace_right

    if total_replace > len(control_epochs):
        raise ValueError("Not enough control epochs to replace the required fraction.")

    ctrl_indices = rng.permutation(len(control_epochs))
    left_ctrl_idx = ctrl_indices[:n_replace_left]
    right_ctrl_idx = ctrl_indices[n_replace_left: n_replace_left + n_replace_right]
    remaining_ctrl_idx = ctrl_indices[n_replace_left + n_replace_right:]
    
    n_keep_left = len(tap_left_epochs) - n_replace_left
    left_keep_idx = rng.choice(len(tap_left_epochs), size=n_keep_left, replace=False)
    new_left = mne.concatenate_epochs([tap_left_epochs[left_keep_idx],
                                       control_epochs[left_ctrl_idx]])
    new_left = new_left[rng.permutation(len(new_left))]

    n_keep_right = len(tap_right_epochs) - n_replace_right
    right_keep_idx = rng.choice(len(tap_right_epochs), size=n_keep_right, replace=False)
    new_right = mne.concatenate_epochs([tap_right_epochs[right_keep_idx],
                                        control_epochs[right_ctrl_idx]])
    new_right = new_right[rng.permutation(len(new_right))]
    
    new_control = control_epochs[remaining_ctrl_idx]
    
    return new_left, new_right, new_control

def main():
    # --- MAIN ---
    time_window = (0, 13)
    subjects = [0, 1, 2, 3, 4]
    fractions = np.arange(0.0, 0.45 + 0.05, 0.05)
    CHANNEL_MODE = "all"  # Options: "inverted", "same_side", or "all"

    results_total = []

    for subj in subjects:
        epochs, evoked_dict = load_data(subject=subj, time_window=(0,11))
        
        for frac in fractions:
            for seed in range(10):
                tap_left_mixed, tap_right_mixed, new_control = replace_fraction_with_control_no_overlap_both(
                    epochs["Tapping_Left"], epochs["Tapping_Right"], epochs["Control"], frac, seed=seed)
                
                # For left hemisphere tapping, use tapping_side "left"
                res_left_hbo  = combined_test(tap_left_mixed,  new_control, (0,11), pick="hbo", seed=seed,
                                            tapping_side="left", channel_mode=CHANNEL_MODE)
                res_left_hbr  = combined_test(tap_left_mixed,  new_control, (0,11), pick="hbr", seed=seed,
                                            tapping_side="left", channel_mode=CHANNEL_MODE)
                # For right hemisphere tapping, use tapping_side "right"
                res_right_hbo = combined_test(tap_right_mixed, new_control, (0,11), pick="hbo", seed=seed,
                                            tapping_side="right", channel_mode=CHANNEL_MODE)
                res_right_hbr = combined_test(tap_right_mixed, new_control, (0,11), pick="hbr", seed=seed,
                                            tapping_side="right", channel_mode=CHANNEL_MODE)
                
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
                results_total.append(result_row)
                
                print(f"\n Subject {subj}, Fraction {frac:.2f}, Seed {seed}, smallest comb p = {smallest_combined:.4e}, smallest overall p = {smallest_overall:.4e} \n")

    df_total = pd.DataFrame(results_total)
    total_csv = "combined_all_results.csv"
    df_total.to_csv(total_csv, index=False)
    print(f"Total results saved to {total_csv}")

    summary = (df_total
            .groupby(["subject", "CONTROL_REPLACEMENT_FRAC"])["smallest_combined_p"]
            .agg(['mean', 'std'])
            .rename(columns={'mean': 'smallest_combined_p_mean', 'std': 'smallest_combined_p_std'})
            .reset_index())

    for subj in summary["subject"].unique():
        subj_summary = summary[summary["subject"] == subj].sort_values(by="CONTROL_REPLACEMENT_FRAC")
        csv_filename = f"subject_{subj}_control_replacement_summary.csv"
        subj_summary.to_csv(csv_filename, index=False)
        print(f"Summary results for subject {subj} saved to {csv_filename}")

if __name__ == "__main__":
    main()
    print("All processing completed.")