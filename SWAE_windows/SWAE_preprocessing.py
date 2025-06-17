"""
Create sparse C×T matrices from sliding windows.

Each sample is all-zeros except for a WINDOW_SIZE slice that preserves the
channel order and temporal position taken from the original epoch.
"""
from itertools import chain
from collections import defaultdict
import numpy as np
from numpy.random import default_rng

# ────────────────────────────────────────────────────────────────────────────
# channel ordering helper (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def _channel_order(epochs_dict):
    left_hbo, right_hbo, mid_hbo = [], [], []
    left_hbr, right_hbr, mid_hbr = [], [], []
    ref = epochs_dict[next(iter(epochs_dict))]

    for idx, ch in enumerate(ref.info["chs"]):
        name = ch["ch_name"].lower()
        x    = None if ch.get("loc") is None else ch["loc"][0]
        side = "mid" if (x is None or np.isclose(x, 0)) else ("left" if x < 0 else "right")

        if "hbo" in name:
            {"left": left_hbo, "right": right_hbo, "mid": mid_hbo}[side].append(idx)
        elif "hbr" in name:
            {"left": left_hbr, "right": right_hbr, "mid": mid_hbr}[side].append(idx)

    return left_hbo + right_hbo + mid_hbo + left_hbr + right_hbr + mid_hbr


# ────────────────────────────────────────────────────────────────────────────
# sliding-window → sparse matrix helper
# ────────────────────────────────────────────────────────────────────────────
def _epoch_to_windows(epoch_arr, window_size, window_step):
    """
    epoch_arr : (C, T) ndarray
    Yields C×T matrices that are zero except for the active window
    """
    C, T = epoch_arr.shape
    for start in range(0, T - window_size + 1, window_step):
        out = np.zeros_like(epoch_arr)
        out[:, start : start + window_size] = epoch_arr[:, start : start + window_size]
        yield out


# ────────────────────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────────────────────
def preprocess_for_wae_windows(epochs_dict,
                               class_labels,
                               window_size,
                               window_step):
    """
    Returns
    -------
    dict {label: [ sample₀, sample₁, … ] }
        each sample is a (C, T) float32 matrix
    """
    out   = defaultdict(list)
    order = _channel_order(epochs_dict)

    for lbl in class_labels:
        data = epochs_dict[lbl].get_data()[:, order, :]    # (E, C, T)
        for ep in data:                                    # (C, T)
            out[lbl].extend(_epoch_to_windows(ep, window_size, window_step))

    # cast once to float32 to save memory
    for lbl in out:
        out[lbl] = [x.astype(np.float32) for x in out[lbl]]
    return out
