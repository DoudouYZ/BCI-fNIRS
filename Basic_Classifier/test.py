import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import mne
from numpy.random import default_rng
from scipy.stats import ttest_rel
from tqdm import tqdm

# Insert parent folder for custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preprocessing.preprocessing_mne import get_raw_subject_data

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def load_data(subject=3, time_window=(-5, 15)):
    tmin = -5.0
    tmax = 15.0
    epochs = get_raw_subject_data(subject=subject, tmin=tmin, tmax=tmax)
    evoked_dict = {
        "Tapping_Left/HbO": epochs["Tapping_Left"].average(picks="hbo"),
        "Tapping_Left/HbR": epochs["Tapping_Left"].average(picks="hbr"),
        "Tapping_Right/HbO": epochs["Tapping_Right"].average(picks="hbo"),
        "Tapping_Right/HbR": epochs["Tapping_Right"].average(picks="hbr"),
        "Control/HbO": epochs["Control"].average(picks="hbo"),
        "Control/HbR": epochs["Control"].average(picks="hbr"),
    }
    for cond in evoked_dict:
        evoked_dict[cond].rename_channels(lambda x: x[:-4])
    return epochs, evoked_dict

def get_channels_by_side(epochs, side):
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

def extract_epoch_means(epochs_obj, time_window, pick_type):
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
    for i in tqdm(range(n_perm), desc=f"Permutations ({pick})", leave=False):
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

def combined_test(epochs_a, epochs_b, time_window, pick, n_perm=5000, seed=42, tapping_side=None):
    """
    If tapping_side is provided, use inverted channel selection.
    """
    if tapping_side is not None:
        opposite_side = "right" if tapping_side.lower() == "left" else "left"
        channels = get_channels_by_side(epochs_a, opposite_side)
        epochs_a = epochs_a.copy().pick(channels)
        epochs_b = epochs_b.copy().pick(channels)
    _, p_phase = permutation_rms_test_pick(epochs_a, epochs_b, pick=pick, time_window=time_window, n_perm=n_perm, seed=seed)
    p_power = average_power_test(epochs_a, epochs_b, time_window, pick)
    p_comb = tipplets_combined_p(p_phase, p_power)
    return {'p_phase': p_phase, 'p_power': p_power, 'p_combined': p_comb}

def combined_test_same(epochs_a, epochs_b, time_window, pick, participant_side, n_perm=5000, seed=42):
    """
    Use same-side channel selection.
    """
    channels = get_channels_by_side(epochs_a, participant_side)
    epochs_a = epochs_a.copy().pick(channels)
    epochs_b = epochs_b.copy().pick(channels)
    _, p_phase = permutation_rms_test_pick(epochs_a, epochs_b, pick=pick, time_window=time_window, n_perm=n_perm, seed=seed)
    p_power = average_power_test(epochs_a, epochs_b, time_window, pick)
    p_comb = tipplets_combined_p(p_phase, p_power)
    return {'p_phase': p_phase, 'p_power': p_power, 'p_combined': p_comb}

def replace_fraction_with_control(tap_epochs, control_epochs, frac, *, seed=123):
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
    mixed = mixed[rng.permutation(len(mixed))]
    return mixed

if __name__ == '__main__':
    SUBJECTS = [0, 1, 2, 3, 4]
    time_window = (0, 13)
    CONTROL_REPLACEMENT_FRAC = 0.2
    PERMUTATIONS = 5000
    seed_list = range(10)

    # Overall accumulator across subjects
    overall_aggregated = {
        "inverted": {"hbo": [], "hbr": [], "combined": []},
        "same_side": {"hbo": [], "hbr": [], "combined": []},
        "all": {"hbo": [], "hbr": [], "combined": []},
    }

    # Loop over each subject.
    for subj in SUBJECTS:
        print(f"\n========== Processing Subject {subj} ==========")
        epochs, evoked_dict = load_data(subject=subj, time_window=time_window)
        aggregated = {
            "inverted": {"hbo": [], "hbr": [], "combined": []},
            "same_side": {"hbo": [], "hbr": [], "combined": []},
            "all": {"hbo": [], "hbr": [], "combined": []},
        }
        for seed in seed_list:
            # Create mixed tapping datasets using the current seed.
            tap_left_mixed = replace_fraction_with_control(epochs["Tapping_Left"], epochs["Control"],
                                                           CONTROL_REPLACEMENT_FRAC, seed=seed)
            tap_right_mixed = replace_fraction_with_control(epochs["Tapping_Right"], epochs["Control"],
                                                            CONTROL_REPLACEMENT_FRAC, seed=seed)

            # --- Inverted configuration ---
            res_left_hbo_inv = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbo",
                                             tapping_side="left", n_perm=PERMUTATIONS, seed=seed)
            res_left_hbr_inv = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbr",
                                             tapping_side="left", n_perm=PERMUTATIONS, seed=seed)
            res_right_hbo_inv = combined_test(tap_right_mixed, epochs["Control"], time_window, pick="hbo",
                                              tapping_side="right", n_perm=PERMUTATIONS, seed=seed)
            res_right_hbr_inv = combined_test(tap_right_mixed, epochs["Control"], time_window, pick="hbr",
                                              tapping_side="right", n_perm=PERMUTATIONS, seed=seed)
            # Calculate min as in your example.
            combined_p_vals_inv = [
                res_left_hbo_inv['p_combined'], res_left_hbr_inv['p_combined'],
                res_right_hbo_inv['p_combined'], res_right_hbr_inv['p_combined']
            ]
            min_inv = min(combined_p_vals_inv)
            # For individual picks, take the min from left & right separately.
            min_inv_hbo = min(res_left_hbo_inv['p_combined'], res_right_hbo_inv['p_combined'])
            min_inv_hbr = min(res_left_hbr_inv['p_combined'], res_right_hbr_inv['p_combined'])
            aggregated["inverted"]["hbo"].append(min_inv_hbo)
            aggregated["inverted"]["hbr"].append(min_inv_hbr)
            aggregated["inverted"]["combined"].append(min_inv)

            # --- Same-side configuration ---
            res_left_hbo_same = combined_test_same(tap_left_mixed, epochs["Control"], time_window, pick="hbo",
                                                   participant_side="left", n_perm=PERMUTATIONS, seed=seed)
            res_left_hbr_same = combined_test_same(tap_left_mixed, epochs["Control"], time_window, pick="hbr",
                                                   participant_side="left", n_perm=PERMUTATIONS, seed=seed)
            res_right_hbo_same = combined_test_same(tap_right_mixed, epochs["Control"], time_window, pick="hbo",
                                                    participant_side="right", n_perm=PERMUTATIONS, seed=seed)
            res_right_hbr_same = combined_test_same(tap_right_mixed, epochs["Control"], time_window, pick="hbr",
                                                    participant_side="right", n_perm=PERMUTATIONS, seed=seed)
            combined_p_vals_same = [
                res_left_hbo_same['p_combined'], res_left_hbr_same['p_combined'],
                res_right_hbo_same['p_combined'], res_right_hbr_same['p_combined']
            ]
            min_same = min(combined_p_vals_same)
            min_same_hbo = min(res_left_hbo_same['p_combined'], res_right_hbo_same['p_combined'])
            min_same_hbr = min(res_left_hbr_same['p_combined'], res_right_hbr_same['p_combined'])
            aggregated["same_side"]["hbo"].append(min_same_hbo)
            aggregated["same_side"]["hbr"].append(min_same_hbr)
            aggregated["same_side"]["combined"].append(min_same)

            # --- All channels configuration ---
            res_left_hbo_all = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbo",
                                             n_perm=PERMUTATIONS, seed=seed)
            res_left_hbr_all = combined_test(tap_left_mixed, epochs["Control"], time_window, pick="hbr",
                                             n_perm=PERMUTATIONS, seed=seed)
            res_right_hbo_all = combined_test(tap_right_mixed, epochs["Control"], time_window, pick="hbo",
                                              n_perm=PERMUTATIONS, seed=seed)
            res_right_hbr_all = combined_test(tap_right_mixed, epochs["Control"], time_window, pick="hbr",
                                              n_perm=PERMUTATIONS, seed=seed)
            combined_p_vals_all = [
                res_left_hbo_all['p_combined'], res_left_hbr_all['p_combined'],
                res_right_hbo_all['p_combined'], res_right_hbr_all['p_combined']
            ]
            min_all = min(combined_p_vals_all)
            min_all_hbo = min(res_left_hbo_all['p_combined'], res_right_hbo_all['p_combined'])
            min_all_hbr = min(res_left_hbr_all['p_combined'], res_right_hbr_all['p_combined'])
            aggregated["all"]["hbo"].append(min_all_hbo)
            aggregated["all"]["hbr"].append(min_all_hbr)
            aggregated["all"]["combined"].append(min_all)

        # Print subject-level averages.
        print(f"\nSubject {subj} results:")
        for cfg in aggregated:
            avg_hbo = np.mean(aggregated[cfg]["hbo"])
            avg_hbr = np.mean(aggregated[cfg]["hbr"])
            avg_combined = np.mean(aggregated[cfg]["combined"])
            print(f"{cfg:10s} - HbO avg: {avg_hbo:.4g}, HbR avg: {avg_hbr:.4g}, Combined avg: {avg_combined:.4g}")
            overall_aggregated[cfg]["hbo"].extend(aggregated[cfg]["hbo"])
            overall_aggregated[cfg]["hbr"].extend(aggregated[cfg]["hbr"])
            overall_aggregated[cfg]["combined"].extend(aggregated[cfg]["combined"])

    # Perform overall tests across subjects.
    print("\n========== Overall Results Across Subjects ==========")
    overall_avg = {}
    for cfg in overall_aggregated:
        avg_hbo = np.mean(overall_aggregated[cfg]["hbo"])
        avg_hbr = np.mean(overall_aggregated[cfg]["hbr"])
        avg_combined = np.mean(overall_aggregated[cfg]["combined"])
        overall_avg[cfg] = avg_combined
        print(f"{cfg:10s} - HbO avg: {avg_hbo:.4g}, HbR avg: {avg_hbr:.4g}, Combined avg: {avg_combined:.4g}")

    best_config = min(overall_avg, key=overall_avg.get)
    print(f"\nBest configuration based on overall combined average p-value: '{best_config}'")

    # Paired t-tests comparing the best config versus the others across subjects.
    print("\n--- Paired t-tests (Combined) Across Subjects ---")
    for cfg in overall_aggregated:
        if cfg == best_config:
            continue
        t_stat, p_val = ttest_rel(overall_aggregated[best_config]["combined"], overall_aggregated[cfg]["combined"])
        print(f"Comparing '{best_config}' vs '{cfg}' (Combined): t = {t_stat:.4g}, p = {p_val:.4g}")

    print("\n--- Paired t-tests (HbO only) Across Subjects ---")
    for cfg in overall_aggregated:
        if cfg == best_config:
            continue
        t_stat, p_val = ttest_rel(overall_aggregated[best_config]["hbo"], overall_aggregated[cfg]["hbo"])
        print(f"Comparing '{best_config}' vs '{cfg}' (HbO): t = {t_stat:.4g}, p = {p_val:.4g}")

    print("\n--- Paired t-tests (HbR only) Across Subjects ---")
    for cfg in overall_aggregated:
        if cfg == best_config:
            continue
        t_stat, p_val = ttest_rel(overall_aggregated[best_config]["hbr"], overall_aggregated[cfg]["hbr"])
        print(f"Comparing '{best_config}' vs '{cfg}' (HbR): t = {t_stat:.4g}, p = {p_val:.4g}")

    all_signif_combined = all(ttest_rel(overall_aggregated[best_config]["combined"], overall_aggregated[cfg]["combined"])[1] < 0.05 
                                for cfg in overall_aggregated if cfg != best_config)
    all_signif_hbo = all(ttest_rel(overall_aggregated[best_config]["hbo"], overall_aggregated[cfg]["hbo"])[1] < 0.05 
                         for cfg in overall_aggregated if cfg != best_config)
    all_signif_hbr = all(ttest_rel(overall_aggregated[best_config]["hbr"], overall_aggregated[cfg]["hbr"])[1] < 0.05 
                         for cfg in overall_aggregated if cfg != best_config)

    if all_signif_combined and all_signif_hbo and all_signif_hbr:
        print(f"\nConfiguration '{best_config}' is statistically significantly better than all other configurations across combined, HbO, and HbR measures (overall). Use '{best_config}'!")
    else:
        print("\nNo configuration was found to be statistically significantly better than all others across all metrics (overall).")