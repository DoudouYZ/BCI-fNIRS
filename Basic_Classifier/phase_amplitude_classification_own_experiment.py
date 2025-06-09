import os
import numpy as np
from numpy.random import default_rng
from scipy.stats import ttest_rel
from tqdm import tqdm
import mne
from mne.preprocessing.nirs import optical_density, beer_lambert_law

# --- I/O & EPOCHING --------------------------------------------------------

def read_snirf(snirf_path):
    """
    Read a SNIRF file, convert to HbO/HbR, return an MNE Raw object.
    """
    print(f"Loading {snirf_path}")
    raw = mne.io.read_raw_snirf(snirf_path, preload=True)
    raw_od = optical_density(raw)
    raw_hb = beer_lambert_law(raw_od)
    return raw_hb


def make_epochs(raw_hb, tmin=-5.0, tmax=15.0, baseline=(None, 0)):
    """
    Epoch raw_hb around two annotation types:
      '1'          → activation
      'gap_period' → control
    Returns a dict: {'activation': Epochs, 'control': Epochs}
    """
    ann_map = {'1': 1, 'control': 2}
    events, used_id = mne.events_from_annotations(raw_hb, event_id=ann_map)

    # OPTION A: just epoch everything, pick by type later
    all_epochs = mne.Epochs(
        raw_hb,
        events,
        event_id=used_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True
    )

    return {
        'activation': all_epochs['1'],
        'control':    all_epochs['control']
    }


# # --- STATISTICAL TESTS -----------------------------------------------------

# def extract_epoch_means(epochs_obj, time_window, pick):
#     data = epochs_obj.get_data(picks=pick)
#     idx = np.where((epochs_obj.times >= time_window[0]) &
#                    (epochs_obj.times <= time_window[1]))[0]
#     return np.mean(np.square(data[:, :, idx]), axis=(1, 2))

# def _rms(x):
#     return np.sqrt(np.mean(np.square(x)))

# def avg_rms_pick(epochs, pick):
#     return _rms(epochs.average(picks=pick).data)

# def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, n_perm=5000, seed=42):
#     rng = default_rng(seed)
#     all_ep = mne.concatenate_epochs([epochs_a, epochs_b])
#     n_a = len(epochs_a)
#     obs = avg_rms_pick(epochs_a, pick) - avg_rms_pick(epochs_b, pick)
#     null = np.empty(n_perm)
#     for i in tqdm(range(n_perm), desc=f"Permutations ({pick})"):
#         idx = rng.permutation(len(all_ep))
#         null[i] = (
#             avg_rms_pick(all_ep[idx[:n_a]], pick)
#           - avg_rms_pick(all_ep[idx[n_a:]], pick)
#         )
#     p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
#     return obs, p

# def average_power_test(epochs_a, epochs_b, time_window, pick):
#     a = extract_epoch_means(epochs_a, time_window, pick)
#     b = extract_epoch_means(epochs_b, time_window, pick)
#     n = min(len(a), len(b))
#     return ttest_rel(a[:n], b[:n]).pvalue

# def tippett_combined_p(p1, p2):
#     return 1 - (1 - min(p1, p2))**2

# def combined_test(epochs_a, epochs_b, time_window, pick, n_perm=5000, seed=42):
#     _, p_phase = permutation_rms_test_pick(
#         epochs_a, epochs_b, pick=pick, n_perm=n_perm, seed=seed
#     )
#     p_power = average_power_test(epochs_a, epochs_b, time_window, pick)
#     return {
#         'p_phase':    p_phase,
#         'p_power':    p_power,
#         'p_combined': tippett_combined_p(p_phase, p_power),
#     }

# def _split_random_half(epochs, *, seed=99):
#     rng = default_rng(seed)
#     idx = rng.permutation(len(epochs))
#     half = len(epochs) // 2
#     return epochs[idx[:half]], epochs[idx[half:half*2]]

# def replace_fraction_with_control(tap_epochs, ctrl_epochs, frac, *, seed=123):
#     rng = default_rng(seed)
#     N = len(tap_epochs)
#     R = int(round(frac * N))
#     if R == 0:
#         return tap_epochs.copy()
#     keep = rng.choice(N, size=N - R, replace=False)
#     take = rng.choice(len(ctrl_epochs), size=R, replace=False)
#     mixed = mne.concatenate_epochs([
#         tap_epochs[keep],
#         ctrl_epochs[take]
#     ])
#     return mixed[rng.permutation(len(mixed))]

# # --- MAIN ------------------------------------------------------------------

# if __name__ == "__main__":
#     # 1) locate your file
#     here      = os.path.dirname(__file__)
#     snirf     = os.path.abspath(os.path.join(here, '..', 'Data', '2_dummy_hand.snirf'))

#     # 2) read & convert
#     raw_hb    = read_snirf(snirf)

#     # 3) epoch activation vs. control
#     epochs    = make_epochs(raw_hb, tmin=-5.0, tmax=15.0)

#     # 4) optional: build evokeds
#     evoked = {}
#     for cond, eps in epochs.items():
#         evoked[f"{cond}/HbO"] = (
#             eps.average(picks='hbo').rename_channels(lambda x: x[:-4])
#         )
#         evoked[f"{cond}/HbR"] = (
#             eps.average(picks='hbr').rename_channels(lambda x: x[:-4])
#         )

#     # 5) mixed vs. control
#     FRAC      = 0.0
#     N_PERM    = 5000
#     print(f"\n--- Mixed Activation (frac={FRAC:.2f}) vs. Control ---")
#     mixed_act = replace_fraction_with_control(epochs['activation'],
#                                               epochs['control'],
#                                               FRAC, seed=42)
#     for pk in ('hbo', 'hbr'):
#         res = combined_test(mixed_act, epochs['control'],
#                             time_window=(0.0, 11.0),
#                             pick=pk, n_perm=N_PERM, seed=42)
#         print(f"{pk.upper()}: combined p={res['p_combined']:.4g} "
#               f"(phase={res['p_phase']:.4g}, power={res['p_power']:.4g})")

#     # 6) control‐split sanity check
#     print("\n--- Control Split Tests ---")
#     for pk in ('hbo', 'hbr'):
#         h1, h2 = _split_random_half(epochs['control'], seed=99)
#         res    = combined_test(h1, h2,
#                                time_window=(0.0, 11.0),
#                                pick=pk, n_perm=N_PERM, seed=42)
#         print(f"{pk.upper()}: combined p={res['p_combined']:.4g} "
#               f"(phase={res['p_phase']:.4g}, power={res['p_power']:.4g})")
# # import os
# import sys
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import mne
# import mne
# from numpy.random import default_rng
# import numpy as np
# from scipy.stats import ttest_rel
# from tqdm import tqdm 
# from mne.preprocessing.nirs import optical_density, beer_lambert_law
# # Insert parent folder for custom modules
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from Preprocessing.preprocessing_mne import get_raw_subject_data

# os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# raw = mne.io.read_raw_snirf("../Data/2_dummy_hand.snirf", preload=True)
# raw_od = optical_density(raw)
# raw_hb = beer_lambert_law(raw_od)
# events, event_id = mne.events_from_annotations(
#     raw_hb,
#     event_id={"activation": 1, "control": 2}
# )

# def read_snirf(snirf_path):
#     """
#     Read a SNIRF file, convert to HbO/HbR, and return a Raw object.
#     """
#     raw = mne.io.read_raw_snirf(snirf_path, preload=True)
#     from mne.preprocessing.nirs import optical_density, beer_lambert_law
#     raw_od = optical_density(raw)
#     raw_hb = beer_lambert_law(raw_od)
#     return raw_hb

# def make_epochs(raw_hb, event_id, tmin=-5.0, tmax=15.0, baseline=(None, 0)):
#     """
#     Epoch a Raw (HbO/HbR) around events in event_id.
#     Returns an Epochs dict keyed by condition.
#     """
#     events, _ = mne.events_from_annotations(raw_hb, event_id=event_id)
#     picks = {"hbo": True, "hbr": True}
#     all_epochs = mne.Epochs(
#         raw_hb, events, event_id=event_id,
#         tmin=tmin, tmax=tmax,
#         picks=picks, baseline=baseline,
#         preload=True
#     )
#     # split into a dict for easy access
#     return {cond: all_epochs[cond] for cond in event_id}


# def load_data(subject=3, time_window=(-5, 15)):
#     """
#     Load data, perform averaging tests and plots for evoked responses,
#     and return concatenated data along with original epoch information.
    
#     This function retains all mean tests and plots of averaged epochs.
#     ICA-related functionality has been removed.
    
#     Returns:
#         tuple: (X, orig_indices, y, control_data, left_data, right_data, numb_samples)
#     """

#     # tmin = time_window[0]
#     # tmax = time_window[1]
#     tmin = -5.0
#     tmax = 15.0

#     picks = {"hbo": True, "hbr": True}

#     # Load subject epochs
#     epochs = mne.Epochs(raw_hb, events, event_id, tmin=tmin, tmax=tmax, pick=picks, baseline=None, preload=True)

#     # Compute evoked responses for each condition and rename channels
#     evoked_dict = {
#         "Tapping_Left/HbO": epochs["Tapping_Left"].average(picks="hbo"),
#         "Tapping_Left/HbR": epochs["Tapping_Left"].average(picks="hbr"),
#         "Tapping_Right/HbO": epochs["Tapping_Right"].average(picks="hbo"),
#         "Tapping_Right/HbR": epochs["Tapping_Right"].average(picks="hbr"),
#         "Control/HbO": epochs["Control"].average(picks="hbo"),
#         "Control/HbR": epochs["Control"].average(picks="hbr"),
#     }
#     for condition in evoked_dict:
#         evoked_dict[condition].rename_channels(lambda x: x[:-4])

#     return epochs, evoked_dict


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

def avg_rms_pick(epochs, pick):
    """
    RMS of the channel-averaged signal using a specific pick.
    """
    evoked = epochs.average(picks=pick)
    return _rms(evoked.data)

def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, n_perm=5000, seed=42):
    """
    Non-parametric permutation test on the RMS of the channel-averaged signal for a given pick.
    Returns the observed difference (A − B) and a two-sided p-value.
    """
    rng = default_rng(seed)
    all_ep = mne.concatenate_epochs([epochs_a, epochs_b])
    n_a = len(epochs_a)
    observed = avg_rms_pick(epochs_a, pick) - avg_rms_pick(epochs_b, pick)
    null_dist = np.empty(n_perm)
    for i in tqdm(range(n_perm), desc=f"Permutations ({pick})"):
        idx = rng.permutation(len(all_ep))
        ep_a = all_ep[idx[:n_a]]
        ep_b = all_ep[idx[n_a:]]
        null_dist[i] = avg_rms_pick(ep_a, pick) - avg_rms_pick(ep_b, pick)
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
    _, p_phase = permutation_rms_test_pick(epochs_a, epochs_b, pick=pick, n_perm=n_perm, seed=seed)
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
script_dir = os.path.dirname(__file__)
snirf_path = os.path.abspath(
    os.path.join(script_dir, '..', 'Data', '3_tongue.snirf')
)
raw_hb = read_snirf(snirf_path)
time_window = (0, 10)
epochs_dict = make_epochs(raw_hb,
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

# --- Run combined tests with the mixed activation data ---------------------
print(f"\n--- Mixed Activation (fraction replaced: {CONTROL_REPLACEMENT_FRAC:.2f}) vs. Control ---")

# HbO
res_hbo = combined_test(
    act_mixed,
    epochs["control"],
    time_window,
    pick="hbo",
    n_perm=PERMUTATIONS
)
print(f"Mixed Activation vs. Control (HbO): combined p = {res_hbo['p_combined']:.4g} "
      f"(phase p = {res_hbo['p_phase']:.4g}, power p = {res_hbo['p_power']:.4g})")

# HbR
res_hbr = combined_test(
    act_mixed,
    epochs["control"],
    time_window,
    pick="hbr",
    n_perm=PERMUTATIONS
)
print(f"Mixed Activation vs. Control (HbR): combined p = {res_hbr['p_combined']:.4g} "
      f"(phase p = {res_hbr['p_phase']:.4g}, power p = {res_hbr['p_power']:.4g})")

# Final p-value
combined_p_values = [
    res_hbo['p_combined'],
    res_hbr['p_combined']
]
min_combined = min(combined_p_values)
print("\nSmallest combined p-value:", min_combined)

if min_combined > 0.05:
    all_p_values = [
        res_hbo['p_phase'],    res_hbo['p_power'],    res_hbo['p_combined'],
        res_hbr['p_phase'],    res_hbr['p_power'],    res_hbr['p_combined']
    ]
    min_all = min(all_p_values)
    print("Since smallest combined p-value > 0.05, smallest overall p-value:", min_all)


# --- Sanity-check: Control split tests ---
print("\n--- Control Split Tests ---")
control_epochs = epochs["control"]   # lowercase 'control'

# For HbO
c_half1, c_half2 = _split_random_half(control_epochs, seed=99)
result_ctrl_hbo = combined_test(
    c_half1,
    c_half2,
    time_window,
    pick="hbo",
    n_perm=PERMUTATIONS
)
print(f"Control Split (HbO): combined p = {result_ctrl_hbo['p_combined']:.4g} "
      f"(phase p = {result_ctrl_hbo['p_phase']:.4g}, power p = {result_ctrl_hbo['p_power']:.4g})")

# For HbR
c_half1, c_half2 = _split_random_half(control_epochs, seed=99)
result_ctrl_hbr = combined_test(
    c_half1,
    c_half2,
    time_window,
    pick="hbr",
    n_perm=PERMUTATIONS
)
print(f"Control Split (HbR): combined p = {result_ctrl_hbr['p_combined']:.4g} "
      f"(phase p = {result_ctrl_hbr['p_phase']:.4g}, power p = {result_ctrl_hbr['p_power']:.4g})")
# =========================  END APPEND  =========================
