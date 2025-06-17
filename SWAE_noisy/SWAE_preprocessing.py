"""
Pre-processing for the new “one-epoch-per-sample” pipeline.
Creates per-label lists of C×T arrays, with optional on-the-fly augmentation.
"""

from pathlib import Path
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────
#  CHANNEL ORDER  (unchanged from earlier)
# ────────────────────────────────────────────────────────────────────────────
def channel_order(epochs_dict):
    left_hbo, right_hbo, mid_hbo = [], [], []
    left_hbr, right_hbr, mid_hbr = [], [], []
    ref = epochs_dict[list(epochs_dict.keys())[0]]

    for idx, ch in enumerate(ref.info["chs"]):
        name = ch["ch_name"].lower()
        loc  = ch.get("loc");  x = None if loc is None else loc[0]

        if x is None or np.isclose(x, 0.0):  side = "mid"
        elif x < 0:                          side = "left"
        else:                                side = "right"

        if   "hbo" in name: bucket = {"left": left_hbo, "right": right_hbo, "mid": mid_hbo}[side]
        elif "hbr" in name: bucket = {"left": left_hbr, "right": right_hbr, "mid": mid_hbr}[side]
        else:               continue
        bucket.append(idx)

    return left_hbo + right_hbo + mid_hbo + left_hbr + right_hbr + mid_hbr


# ────────────────────────────────────────────────────────────────────────────
#  AUGMENTATION HELPERS
# ────────────────────────────────────────────────────────────────────────────
def _augment_epoch(x, rng):
    """x : (C,T) float32 array  – returns a *new* augmented copy."""
    C, T = x.shape
    # 1. additive jitter
    x = x + rng.normal(0.0, 2e-6, size=x.shape)
    # 2. amplitude scaling
    x = x * rng.uniform(0.95, 1.05)
    # 3. ±5 % time-warp
    warp = rng.uniform(0.95, 1.05)
    t_new = int(round(T * warp))
    x = np.stack([np.interp(np.linspace(0, T-1, t_new), np.arange(T), ch)
                  for ch in x], axis=0)
    # crop / pad back
    if t_new >= T:            x = x[:, :T]
    else:                     x = np.pad(x, ((0,0),(0, T-t_new)), 'edge')
    # 4. random HbO/HbR pair dropout
    if rng.random() < 0.1:
        pair = rng.integers(0, C//2) * 2
        x[pair:pair+2] = 0.0
    return x.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
#  PUBLIC FUNCTION
# ────────────────────────────────────────────────────────────────────────────
def preprocess_epochs(epochs_dict, *, n_aug=5, rng_seed=0, zscore=True):
    """
    Returns dict{label: list[np.ndarray (C,T)]}.
    Each original epoch appears once plus `n_aug` augmented copies.
    """
    rng = default_rng(rng_seed)
    order = channel_order(epochs_dict)
    out = {}

    for lbl, ep in epochs_dict.items():
        base = ep.get_data()[:, order, :]           # (E,C,T)
        samples = []
        for epoch in tqdm(base, desc=f"Augmenting {lbl}"):
            # original
            samples.append(epoch.astype(np.float32))
            # copies
            for _ in range(n_aug):
                samples.append(_augment_epoch(epoch.copy(), rng))

        # optional per-channel z-score
        if zscore:
            for i, arr in enumerate(samples):
                m = arr.mean(axis=1, keepdims=True)
                s = arr.std(axis=1, keepdims=True) + 1e-6
                samples[i] = (arr - m) / s
        out[lbl] = samples
    return out
