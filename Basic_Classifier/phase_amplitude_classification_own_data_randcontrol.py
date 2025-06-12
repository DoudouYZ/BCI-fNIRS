import os
import numpy as np
from numpy.random import default_rng
from scipy.stats import ttest_rel
from tqdm import tqdm
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
        baseline=baseline,
        preload=True,
        detrend=None,
        verbose=True,
    )

    return {
        'activation': epochs['1'],
        'control':    epochs['control']
    }

def get_epochs_data(epochs_obj, pick=None):
    import mne
    if isinstance(epochs_obj, list):
        data_list = []
        times_list = []
        for e in epochs_obj:
            # Each control epoch is a single-epoch EpochsArray.
            data = e.get_data()[0]  # shape: (n_channels, n_times)
            if pick is not None:
                # Select channels whose names contain the pick string (case-insensitive)
                picks_idx = [i for i, ch in enumerate(e.info['ch_names']) if pick.lower() in ch.lower()]
                data = data[picks_idx, :]
            data_list.append(data)
            times_list.append(e.times)
        # Find the shortest time axis length so we can crop all epochs to the same number of samples
        min_len = min(d.shape[1] for d in data_list)
        data_list = [d[:, :min_len] for d in data_list]
        data_arr = np.stack(data_list, axis=0)
        times = times_list[0][:min_len]
        return data_arr, times
    else:
        if pick is not None:
            data_arr = epochs_obj.get_data(picks=pick)
        else:
            data_arr = epochs_obj.get_data()
        # Crop to full length as the common time axis
        min_len = data_arr.shape[2]
        times = epochs_obj.times[:min_len]
        return data_arr, times
    

# def make_epochs_rand_control(raw_hb, baseline=(None, 0)):
#     """
#     Epoch raw_hb around two annotation types:
#       '1'          → activation (fixed time window)
#       'control'    → control (random time window with same duration as activation)
#     Returns a dict: {'activation': Epochs, 'control': Epochs}
#     """
#     ann_map = {'1': 1, 'control': 2}
#     events, used_id = mne.events_from_annotations(raw_hb, event_id=ann_map)

#     # Fixed time window for activation epochs.
#     activation_tmin = -5.0
#     activation_tmax = 15.0
#     duration = activation_tmax - activation_tmin  # duration to be preserved

#     # Randomly sample control epoch window.
#     rng = np.random.default_rng()
#     control_tmin = rng.uniform(-10.0, 5.0)  # for example, anywhere in [-7, -1]
#     control_tmax = control_tmin + duration

#     # Define rejection criteria.
#     reject_criteria = dict(hbo=80e-6)

#     # Create activation epochs with fixed parameters.
#     epochs_activation = mne.Epochs(
#         raw_hb,
#         events,
#         event_id={'1': used_id['1']},
#         tmin=activation_tmin,
#         tmax=activation_tmax,
#         reject=reject_criteria,
#         reject_by_annotation=True,
#         proj=True,
#         baseline=baseline,
#         preload=True,
#         detrend=None,
#         verbose=True,
#     )

#     # Create control epochs with randomly sampled tmin and tmax.
#     epochs_control = mne.Epochs(
#         raw_hb,
#         events,
#         event_id={'control': used_id['control']},
#         tmin=control_tmin,
#         tmax=control_tmax,
#         reject=reject_criteria,
#         reject_by_annotation=True,
#         proj=True,
#         baseline=(None, control_tmin+5.0),
#         preload=True,
#         detrend=None,
#         verbose=True,
#     )

#     # Make sure both epochs have the same time axis.
#     if len(epochs_activation.times) != len(epochs_control.times):
#         min_len = min(len(epochs_activation.times), len(epochs_control.times))
#         # Crop activation epochs if needed
#         epochs_activation._data = epochs_activation.get_data()[:, :, :min_len]
#         epochs_activation._times = epochs_activation.times[:min_len]
#         # Also crop control epochs for consistency (if needed)
#         epochs_control._data = epochs_control.get_data()[:, :, :min_len]
#         epochs_control._times = epochs_control.times[:min_len]
#         print(f"Length of activation epochs ({len(epochs_activation.times)}) ")
#         print(f"length of control epochs ({len(epochs_control.times)}) ")
#     return {
#         'activation': epochs_activation,
#         'control':    epochs_control
#     }


def make_epochs_rand_control(raw_hb, baseline=(None, 0)):
    """
    Create epochs for:
      - Activation: fixed time window.
      - Control: each control event is assigned its own random time window (with the same duration as activation).
    
    This function creates control epochs by iterating over control events,
    sampling a random tmin for that epoch, extracting data from raw_hb and then
    assembling an EpochsArray. Finally, the control epochs are concatenated and
    both activation and control epochs are cropped to have the same time axis.
    
    Returns a dict: {'activation': Epochs, 'control': Epochs}
    """
    ann_map = {'1': 1, 'control': 2}
    events, used_id = mne.events_from_annotations(raw_hb, event_id=ann_map)
    
    # Fixed parameters for activation epochs.
    activation_tmin = -5.0
    activation_tmax = 15.0
    duration = activation_tmax - activation_tmin

    # Create activation epochs normally.
    epochs_activation = mne.Epochs(
        raw_hb,
        events,
        event_id={'1': used_id['1']},
        tmin=activation_tmin,
        tmax=activation_tmax,
        reject=dict(hbo=80e-6),
        reject_by_annotation=True,
        proj=True,
        baseline=baseline,
        preload=True,
        detrend=None,
        verbose=True,
    )

    # Select control events.
    ctrl_mask = events[:, 2] == used_id['control']
    events_control = events[ctrl_mask]
    
    rng = np.random.default_rng()
    control_epochs_list = []
    for ev in events_control:
        # Event onset in seconds.
        event_time = ev[0] / raw_hb.info['sfreq']
        # Sample a random window for this control epoch.
        # You can adjust the bounds (here -10.0 and 5.0) as needed.
        tmin_i = rng.uniform(-20.0, -5.0)  
        tmax_i = tmin_i + duration
        baseline = (None, tmin_i + 5.0)  # Baseline relative to the control epoch's tmin
        # Crop raw_hb to the desired window relative to the event onset.
        # This produces a Raw object containing only the desired time segment.
        raw_crop = raw_hb.copy().crop(tmin=event_time + tmin_i, tmax=event_time + tmax_i)
        data = raw_crop.get_data()
        # Create an EpochsArray for this one epoch.
        # The new epoch's time vector will start at tmin_i.
        epoch_i = mne.EpochsArray(data[np.newaxis, ...], raw_hb.info, tmin=tmin_i, baseline=baseline)
        control_epochs_list.append(epoch_i)
    
    # Concatenate all control epochs.
    # epochs_control = mne.concatenate_epochs(control_epochs_list)
    
    # # Crop both activation and control epochs so that their time axes have the same length.
    # min_len = min(len(epochs_activation.times), len(epochs_control.times))
    # epochs_activation._data = epochs_activation.get_data()[:, :, :min_len]
    # epochs_activation._times = epochs_activation.times[:min_len]
    # epochs_control._data = epochs_control.get_data()[:, :, :min_len]
    # epochs_control._times = epochs_control.times[:min_len]
    
    return {'activation': epochs_activation, 'control': control_epochs_list}




# # --- STATISTICAL TESTS -----------------------------------------------------
def extract_epoch_means(epochs_obj, time_window, pick):
    """
    Compute the mean power for each epoch over channels and a time window.
    """
    data, times = get_epochs_data(epochs_obj, pick=pick)
    idx = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]
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

# def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
#     """
#     Non-parametric permutation test on the RMS of the channel-averaged signal for a given pick.
#     Returns the observed difference (A − B) and a two-sided p-value.
#     """
#     rng = default_rng(seed)
#     all_ep = mne.concatenate_epochs([epochs_a, epochs_b])
#     n_a = len(epochs_a)
#     observed = avg_rms_pick(epochs_a, pick, time_window) - avg_rms_pick(epochs_b, pick, time_window)
#     null_dist = np.empty(n_perm)
#     for i in tqdm(range(n_perm), desc=f"Permutations ({pick})"):
#         idx = rng.permutation(len(all_ep))
#         ep_a = all_ep[idx[:n_a]]
#         ep_b = all_ep[idx[n_a:]]
#         null_dist[i] = avg_rms_pick(ep_a, pick, time_window) - avg_rms_pick(ep_b, pick, time_window)
#     p_val = (np.sum(np.abs(null_dist) >= abs(observed)) + 1) / (n_perm + 1)
#     return observed, p_val

def permutation_rms_test_pick(epochs_a, epochs_b, *, pick, time_window, n_perm=5000, seed=42):
    rng = default_rng(seed)
    # Get data arrays and times for both groups.
    data_a, times_a = get_epochs_data(epochs_a, pick=pick)
    data_b, times_b = get_epochs_data(epochs_b, pick=pick)
    # Crop time axis to the minimum length across both groups.
    min_t = min(data_a.shape[2], data_b.shape[2])
    data_a = data_a[:, :, :min_t]
    data_b = data_b[:, :, :min_t]
    times = times_a[:min_t]

    n_a = data_a.shape[0]
    n_total = data_a.shape[0] + data_b.shape[0]

    def compute_avg_rms(data, times, time_window):
        # Compute the average (evoked) across epochs.
        evoked = np.mean(data, axis=0)  # shape: (n_channels, n_times)
        idx = np.where((times >= time_window[0]) & (times <= time_window[1]))[0]
        return np.sqrt(np.mean(evoked[:, idx] ** 2))

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

def _split_random_half(epochs, *, seed=99):
    rng = default_rng(seed)
    if isinstance(epochs, list):
        indices = rng.permutation(len(epochs))
        half = len(epochs) // 2
        half1 = [epochs[i] for i in indices[:half]]
        half2 = [epochs[i] for i in indices[half:half*2]]
        return half1, half2
    else:
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
    Return a new "epochs" (in list form) in which a fraction (`frac`) of tap_epochs
    has been replaced by randomly-selected epochs from control_epochs.
    """
    import mne
    assert 0.0 <= frac <= 1.0, "Fraction must be between 0 and 1."
    rng = default_rng(seed)
    n_total = len(tap_epochs)
    n_replace = int(round(frac * n_total))
    if n_replace == 0:
        # Already a list? Otherwise convert.
        if isinstance(tap_epochs, list):
            return tap_epochs.copy()
        else:
            return [mne.EpochsArray(d[np.newaxis, ...], tap_epochs.info, tmin=tap_epochs.tmin)
                    for d in tap_epochs.get_data()]
    
    n_keep = n_total - n_replace
    tap_keep_idx = rng.choice(n_total, size=n_keep, replace=False)
    ctrl_idx = rng.choice(len(control_epochs), size=n_replace, replace=False)
    
    # Convert tap_epochs (which is an mne Epochs object) into a list of single-epoch objects.
    tap_list = [mne.EpochsArray(d[np.newaxis, ...], tap_epochs.info, tmin=tap_epochs.tmin)
                for d in tap_epochs.get_data()]
    
    mixed = [tap_list[i] for i in tap_keep_idx] + [control_epochs[i] for i in ctrl_idx]
    rng.shuffle(mixed)
    return mixed

# --- MAIN ---
script_dir = os.path.dirname(__file__)
snirf_path = os.path.abspath(
    os.path.join(script_dir, '..', 'Data', '2_tongue.snirf')
)

raw_haemo = read_snirf(snirf_path)
time_window = (0, 13)
# epochs_dict = make_epochs(raw_haemo)
epochs_dict = make_epochs_rand_control(raw_haemo)
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
