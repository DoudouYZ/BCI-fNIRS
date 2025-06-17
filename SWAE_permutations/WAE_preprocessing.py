from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
from itertools import combinations, islice
from collections import defaultdict

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def epochs_to_2d(epochs):
    """
    Convert an mne.Epochs object into a 2-D NumPy array with shape
    (C·T, E), where rows iterate over time inside each channel and
    columns correspond to epochs.

    Row ordering → ch0:t0..tN, ch1:t0..tN, …, chC−1:t0..tN
    """
    data = epochs.get_data()           # (E, C, T)
    E, C, T = data.shape
    # stack channel blocks vertically (time is fastest axis)
    mat = np.vstack([data[:, c, :].T for c in range(C)])  # (C·T, E)
    return mat, C, T

def generate_permutation_matrices(mat, C, T, n_perm, max_perm=None, rng=None):
    """
    Create channel-wise permutation samples.

    Yields, for each unique (randomised) subset of n_perm epochs:
        list[ np.ndarray ]  of length C
            each entry has shape (T, n_perm)
    """
    rng = np.random.default_rng(rng)
    E = mat.shape[1]
    if n_perm > E:
        raise ValueError(f"n_perm={n_perm} > available epochs ({E})")

    idx = list(combinations(range(E), n_perm))
    rng.shuffle(idx)
    if max_perm is not None:
        idx = idx[:max_perm]

    for cols in tqdm(idx, desc="Generating permutation matrices", total=len(idx)):
        sub = mat[:, cols]                      # (C·T, n_perm)
        yield [sub[c*T:(c+1)*T, :] for c in range(C)]


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------
def preprocess_for_wae(epochs, class_labels, n_perm=4, max_perm=None, rng=None):
    """
    Build per-class datasets ready for WAE training.

    Returns
    -------
    dict
        {label: list_of_samples}
        where each sample is a list with C arrays of shape (T, n_perm)
    """
    out = {}
    for lbl in class_labels:
        ep          = epochs[lbl]
        base2d, C,T = epochs_to_2d(ep)
        out[lbl]    = list(
            generate_permutation_matrices(base2d, C, T,
                                          n_perm=n_perm,
                                          max_perm=max_perm,
                                          rng=rng)
        )
    return out


def _channel_order(epochs):
    """
    Return an index array that re-orders channels to:
        [left HbO, right HbO, center HbO,
         left HbR, right HbR, center HbR]
    Unknown/centre channels come last in each chromophore block.
    """
    left_hbo, right_hbo, mid_hbo = [], [], []
    left_hbr, right_hbr, mid_hbr = [], [], []

    ref = epochs[list(epochs.keys())[0]]
    for idx, ch in enumerate(ref.info["chs"]):
        name = ch["ch_name"].lower()
        loc  = ch.get("loc")
        x    = None if loc is None else loc[0]

        # decide side
        if x is None or np.isclose(x, 0.0):
            side = "mid"
        elif x < 0:
            side = "left"
        else:
            side = "right"

        # decide chromophore
        if "hbo" in name:
            bucket = {"left": left_hbo, "right": right_hbo, "mid": mid_hbo}[side]
        elif "hbr" in name:
            bucket = {"left": left_hbr, "right": right_hbr, "mid": mid_hbr}[side]
        else:                       # ignore short-separation/dark channels
            continue

        bucket.append(idx)

    # concatenate into final order
    return (
        left_hbo + right_hbo + mid_hbo +
        left_hbr + right_hbr + mid_hbr
    )


def _epoch_cube_list(data, n_perm, max_perm=None, rng=None):
    """
    Parameters
    ----------
    data : ndarray (E, C, T)  – epochs already channel-reordered
    n_perm : int              – epochs per cube
    max_perm : int|None       – cap on number of cubes (after shuffling)
    rng : np.random.Generator

    Yields
    ------
    cube : ndarray (n_perm, C, T)
    """
    rng = default_rng(rng)
    E   = data.shape[0]
    if n_perm > E:
        raise ValueError("n_perm larger than available epochs.")

    combos = list(combinations(range(E), n_perm))
    rng.shuffle(combos)
    if max_perm is not None:
        combos = combos[:max_perm]

    for idxs in combos:
        yield data[np.array(idxs), :, :]


def preprocess_for_wae_cube(epochs, class_labels,
                            n_perm=3, max_perm=None, rng=None):
    """
    Build permutation cubes for every requested label.

    Returns
    -------
    dict  {label: [cube₀, cube₁, …]}
        Each cube has shape (n_perm, C, T) with consistent
        channel order defined by `_channel_order`.
    """
    out = {}
    ch_order = _channel_order(epochs)         # same order for all labels
    for lbl in class_labels:
        ep = epochs[lbl]
        dat = ep.get_data()[:, ch_order, :]   # (E, C, T)
        cubes = list(_epoch_cube_list(dat, n_perm,
                                      max_perm=max_perm, rng=rng))
        out[lbl] = cubes
    return out

