"""
save_results.py
Run VAE committees, bucket the outputs, and save everything to .npz.
Works with the new committee_for_subject that returns
   {"right": {...}, "left": {...}}
or {"all_control": {...}} when ALL_CONTROL=True.
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# -------------------------------------------------
# Save everything (robust)
# -------------------------------------------------
def _to_nparray(v):
    """Ensure v is a NumPy array; use dtype=object if ragged."""
    if isinstance(v, np.ndarray):
        return v
    try:                       # will succeed for numeric 1-D lists
        return np.asarray(v)
    except ValueError:         # ragged → force object array
        return np.asarray(v, dtype=object)


# ------------------------------------------------------------------
# make project root importable
# ------------------------------------------------------------------
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from VAE_committee import committee_for_subject


# ------------------------------------------------------------------
# helper to bucket samples and compute interval means
# ------------------------------------------------------------------
def post_process(res):
    """Return dict with added samples_{ctrl,left,right} and interval_means."""
    consensus = res["consensus"]
    times     = res["times"]
    events    = np.sort(res["events"])
    raw       = res["raw"]

    onsets = raw.annotations.onset
    descs  = np.asarray(raw.annotations.description)
    names  = [descs[np.where(np.isclose(onsets, ev))[0][0]]
              if np.any(np.isclose(onsets, ev)) else "Unknown"
              for ev in events]

    ctl, tl, tr = [], [], []
    for (s, e), name in zip(zip(events[:-1], events[1:]), names[:-1]):
        mask = (times >= s) & (times < e)
        if not mask.any():
            continue
        vals = consensus[mask]
        if "Control" in name:
            ctl.extend(vals)
        elif "Tapping_Left" in name:
            tl.extend(vals)
        elif "Tapping_Right" in name:
            tr.extend(vals)

    # epoch means
    interval_means = [consensus[(times >= s) & (times < e)].mean()
                      for s, e in zip(events[:-1], events[1:])]

    res.update(dict(
        samples_control=np.array(ctl),
        samples_tap_left=np.array(tl),
        samples_tap_right=np.array(tr),
        interval_means=np.array(interval_means),
        event_names=names
    ))
    return res


# ------------------------------------------------------------------
# master function
# ------------------------------------------------------------------
def save_results(participant_idx=4, ALL_CONTROL=False):
    # ---------- hyper‑parameters ----------
    seeds         = tuple(range(1))
    means         = 1.0
    logvar        = 0.0
    beta          = 0.75          # only used if use_mmd=False
    window_length = 80
    window_buffer = 2.3
    latent_dim    = 8
    epochs_num    = 20

    # new WAE knobs (will be ignored if committee use_mmd=False)
    use_mmd   = True
    lam_mmd   = 25
    mmd_sigma = 1.8

    # --------------------------------------
    results_dict = committee_for_subject(
        participant_idx=participant_idx,
        seeds=seeds,
        beta=beta,
        means=means,
        logvar=logvar,
        window_length=window_length,
        window_buffer=window_buffer,
        latent_dim=latent_dim,
        epochs_num=epochs_num,
        use_mmd=use_mmd,          
        lam_mmd=lam_mmd,          
        mmd_sigma=mmd_sigma,      
        verbose=True,
        ALL_CONTROL=ALL_CONTROL,
        debug=False
    )

    # ensure output directory
    out_dir = Path(f"subject_results/results_subject{participant_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # loop over keys: "right"/"left"  or  "all_control"
    for key, res_raw in results_dict.items():
        res = post_process(res_raw)          # add buckets + means

        fname = (f"ALL_CONTROL_subject_{participant_idx}_{len(seeds)}_seeds.npz"
                 if key == "all_control"
                 else f"subject_{participant_idx}_{key}_{len(seeds)}_seeds.npz")
        out_path = out_dir / fname

        res["starts"] = res_raw["starts"]  # or if post_process has access, just leave it in
        # convert ragged lists to object-dtype arrays
        for key in list(res.keys()):
            res[key] = _to_nparray(res[key])
        np.savez(out_path, **res)
        print(f"Saved {out_path}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    IDs = (0,1,2,3,4)
    ctrl = (False, True)
    IDs = (4,)
    ctrl = (False,)
    pbar = tqdm(total=len(IDs)*len(ctrl), desc="saving results")
    for pid in IDs:
        for ctrl_flag in ctrl:
            save_results(participant_idx=pid, ALL_CONTROL=ctrl_flag)
            pbar.update(1)
