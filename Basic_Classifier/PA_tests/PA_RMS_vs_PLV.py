import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import ttest_rel
from scipy.signal import hilbert
from numpy.random import default_rng
from tqdm import tqdm

# Insert parent folder for custom modules if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Preprocessing.preprocessing_mne import get_raw_subject_data

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def load_data(subject=3, time_window=(-5, 15)):
    """
    Load data, perform averaging tests and plots for evoked responses,
    and return epochs and evoked responses.
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

def _rms(x):
    """Root-mean-square of an array."""
    return np.sqrt(np.mean(np.square(x)))

def avg_rms_pick_window(epochs, pick, time_window):
    """
    RMS of the channel-averaged signal computed within a time window.
    """
    evoked = epochs.average(picks=pick)
    idx = np.where((evoked.times >= time_window[0]) & (evoked.times <= time_window[1]))[0]
    return _rms(evoked.data[:, idx])

def compute_plv(epochs, pick, time_window):
    """
    Compute the Phase-Locking Value (PLV) over a time window.
    First, average across channels then compute the analytic signal.
    The PLV is the average over time of the magnitude of the mean phase vector across epochs.
    """
    data = epochs.get_data(picks=pick)  # shape (n_epochs, n_channels, n_times)
    idx = np.where((epochs.times >= time_window[0]) & (epochs.times <= time_window[1]))[0]
    # Average channels => shape (n_epochs, n_times)
    data_avg = np.mean(data[:, :, idx], axis=1)
    # Compute analytic signal for each trial on the time axis.
    analytic = hilbert(data_avg, axis=1)
    phase = np.angle(analytic)  # shape (n_epochs, n_times)
    # Compute PLV at each time sample across trials.
    plv_time = np.abs(np.mean(np.exp(1j * phase), axis=0))
    return np.mean(plv_time)  # average over the time window

def avg_plv_pick_window(epochs, pick, time_window):
    """
    Wrapper to compute PLV for the channel-averaged power over a given time window.
    """
    return compute_plv(epochs, pick, time_window)

def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
    """
    Permutation test on the RMS measure.
    """
    rng = default_rng(seed)
    all_ep = mne.concatenate_epochs([epochs_a, epochs_b])
    n_a = len(epochs_a)
    observed = avg_rms_pick_window(epochs_a, pick, time_window) - avg_rms_pick_window(epochs_b, pick, time_window)
    null_dist = np.empty(n_perm)
    for i in tqdm(range(n_perm), desc=f"Permutations ({pick} - RMS)"):
        idx = rng.permutation(len(all_ep))
        ep_a = all_ep[idx[:n_a]]
        ep_b = all_ep[idx[n_a:]]
        null_dist[i] = avg_rms_pick_window(ep_a, pick, time_window) - avg_rms_pick_window(ep_b, pick, time_window)
    p_val = (np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p_val

def permutation_plv_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
    """
    Permutation test on the PLV measure.
    """
    rng = default_rng(seed)
    all_ep = mne.concatenate_epochs([epochs_a, epochs_b])
    n_a = len(epochs_a)
    observed = avg_plv_pick_window(epochs_a, pick, time_window) - avg_plv_pick_window(epochs_b, pick, time_window)
    null_dist = np.empty(n_perm)
    for i in tqdm(range(n_perm), desc=f"Permutations ({pick} - PLV)"):
        idx = rng.permutation(len(all_ep))
        ep_a = all_ep[idx[:n_a]]
        ep_b = all_ep[idx[n_a:]]
        null_dist[i] = avg_plv_pick_window(ep_a, pick, time_window) - avg_plv_pick_window(ep_b, pick, time_window)
    p_val = (np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p_val

def average_power_test(epochs_a, epochs_b, time_window, pick):
    """
    Paired t-test for average power per epoch.
    """
    data_a = extract_epoch_means(epochs_a, time_window, pick)
    data_b = extract_epoch_means(epochs_b, time_window, pick)
    # Equalize trial counts:
    n = min(len(data_a), len(data_b))
    data_a, data_b = data_a[:n], data_b[:n]
    _, p_val = ttest_rel(data_a, data_b)
    return p_val

def tipplets_combined_p(p1, p2):
    """
    Combine two p-values using Tippett's method.
    """
    return 1 - (1 - min(p1, p2))**2

def combined_test(epochs_a, epochs_b, time_window, pick, phase_method="rms", n_perm=5000, seed=42):
    """
    Runs both the phase-alignment permutation test (using either RMS or PLV) 
    and the average power test.
    phase_method: "rms" or "plv"
    Returns a dictionary with p-values.
    """
    if phase_method == "rms":
        _, p_phase = permutation_rms_test_pick(epochs_a, epochs_b, pick=pick, time_window=time_window, n_perm=n_perm, seed=seed)
    elif phase_method == "plv":
        _, p_phase = permutation_plv_test_pick(epochs_a, epochs_b, pick=pick, time_window=time_window, n_perm=n_perm, seed=seed)
    else:
        raise ValueError("Unknown phase method. Choose 'rms' or 'plv'.")
    p_power = average_power_test(epochs_a, epochs_b, time_window, pick)
    p_comb = tipplets_combined_p(p_phase, p_power)
    return {'p_phase': p_phase, 'p_power': p_power, 'p_combined': p_comb}

def _split_random_half(epochs, *, seed=99):
    """Randomly split an Epochs object into two equal halves."""
    rng = default_rng(seed)
    idx = rng.permutation(len(epochs))
    half = len(epochs) // 2
    return epochs[idx[:half]], epochs[idx[half:half*2]]

def replace_fraction_with_control(tap_epochs, control_epochs, frac, *, seed=123):
    """
    Replace a fraction of tapping epochs with control epochs.
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
    mixed = mixed[rng.permutation(len(mixed))]
    return mixed

def main():
    parser = argparse.ArgumentParser(description="Run phase and power tests using different phase methods.")
    parser.add_argument("--subject", type=int, default=2, help="Subject identifier (default: 2)")
    parser.add_argument("--time_window", type=float, nargs=2, default=[0, 13],
                        help="Time window (start end) in seconds (default: 0 13)")
    parser.add_argument("--pick", type=str, default="hbo", help="Pick type (e.g. 'hbo' or 'hbr')")
    parser.add_argument("--phase_method", type=str, choices=["rms", "plv"], default="rms",
                        help="Phase method: 'rms' or 'plv' (default: rms)")
    parser.add_argument("--perm", type=int, default=5000, help="Number of permutations (default: 5000)")
    parser.add_argument("--control_frac", type=float, default=0.0, help="Fraction of tapping epochs to replace with control (default: 0.0)")

    # If running from VS Code (or without arguments), provide default args.
    if len(sys.argv) == 1:
        sys.argv.extend(["--subject", "2", "--time_window", "0", "13",
                         "--pick", "hbo", "--phase_method", "rms",
                         "--perm", "5000", "--control_frac", "0.5"])
    args = parser.parse_args()

    subject = args.subject
    time_window = tuple(args.time_window)
    pick = args.pick
    phase_method = args.phase_method
    n_perm = args.perm
    control_frac = args.control_frac

    print(f"Running tests for subject {subject}, time window {time_window}, pick '{pick}', phase method '{phase_method}'")

    epochs, evoked_dict = load_data(subject=subject, time_window=time_window)

    # Create mixed tapping datasets if required
    tap_left = epochs["Tapping_Left"]
    tap_right = epochs["Tapping_Right"]
    control = epochs["Control"]
    tap_left_mixed  = replace_fraction_with_control(tap_left, control, control_frac, seed=42)
    tap_right_mixed = replace_fraction_with_control(tap_right, control, control_frac, seed=42)

    print(f"\n--- Mixed Tapping (fraction replaced: {control_frac:.2f}) vs. Control ---")
    res_left = combined_test(tap_left_mixed, control, time_window, pick, phase_method=phase_method, n_perm=n_perm)
    print(f"Tapping_Left vs. Control: combined p = {res_left['p_combined']:.4g} "
          f"(phase p = {res_left['p_phase']:.4g}, power p = {res_left['p_power']:.4g})")
    res_right = combined_test(tap_right_mixed, control, time_window, pick, phase_method=phase_method, n_perm=n_perm)
    print(f"Tapping_Right vs. Control: combined p = {res_right['p_combined']:.4g} "
          f"(phase p = {res_right['p_phase']:.4g}, power p = {res_right['p_power']:.4g})")

    # Sanity-check using Control split tests
    print("\n--- Control Split Tests ---")
    c_half1, c_half2 = _split_random_half(control, seed=99)
    res_ctrl = combined_test(c_half1, c_half2, time_window, pick, phase_method=phase_method, n_perm=n_perm)
    print(f"Control Split: combined p = {res_ctrl['p_combined']:.4g} "
          f"(phase p = {res_ctrl['p_phase']:.4g}, power p = {res_ctrl['p_power']:.4g})")

if __name__ == "__main__":
    main()