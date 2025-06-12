"""
vae_committee.py
Utility to train a committee (ensemble) of Mixture-VAEs on a single
subject’s continuous fNIRS data and return a consensus label trace.

Requires:
  • MixtureVAE, train_mixture_vae, create_sliding_windows_no_classes,
    per_timepoint_labels_sparse  (from your Classifier / Preprocessing code)
  • get_continuous_subject_data  (from your preprocessing pipeline)

Author:  (re-created from lost file)
"""
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import gc
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mne

# -------------------------------------------------------------------------
# Import your own project modules here
# -------------------------------------------------------------------------
from Classifier.AE_models import (
    MixtureVAE,
    train_mixture_vae,
    create_sliding_windows_no_classes,
    per_timepoint_labels_sparse,
)
from Preprocessing import get_continuous_subject_data
import matplotlib.pyplot as plt

def load_and_preprocess(participant_idx, all_control=False,
                        fake_cycle=("Control", "Tapping_Left", "Tapping_Right")):
    """
    Load continuous data for one subject.

    If `all_control` is False  -> return data + original events.
    If `all_control` is True   -> keep only control epochs, relabel them
                                 Control → Tap-L → Tap-R → … cyclically,
                                 and return the modified events array.

    Returns
    -------
    X        : ndarray (n_ch, n_times_kept)
    raw      : MNE Raw object (annotations may be overwritten if all_control)
    sfreq    : float
    events   : 1-D ndarray of event onsets (seconds)
    times    : 1-D ndarray of time stamps for kept samples
    """
    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    raw, sfreq, events_orig = get_continuous_subject_data(subject=participant_idx)
    X_full = raw.get_data()           # (n_ch, n_times)
    times_full = raw.times

    if not all_control:
        # -------- ordinary case: keep everything ----------------------
        X = X_full
        times = times_full
        events = events_orig

    else:
        # -------- keep only control epochs ----------------------------
        onsets  = raw.annotations.onset
        descs   = raw.annotations.description
        ctrl_mask = np.zeros(times_full.size, dtype=bool)

        new_event_starts = []
        new_event_labels = []

        cycle_len = len(fake_cycle)
        ctrl_count = 0

        for i, (start, desc) in enumerate(zip(onsets, descs)):
            end = onsets[i + 1] if i + 1 < len(onsets) else times_full[-1] + 1 / sfreq
            if "Control" not in desc:
                continue

            # mark mask
            m = (times_full >= start) & (times_full < end)
            ctrl_mask |= m

            # record new event onset and label
            new_event_starts.append(start)
            label = fake_cycle[ctrl_count % cycle_len]
            new_event_labels.append(label)
            ctrl_count += 1

        # close with trailing dummy
        events = np.array(new_event_starts + [times_full[-1] + 1 / sfreq])

        # fake_events : 1-D array of N+1 onsets   (sec)
        onsets       = np.array(new_event_starts)        # length N
        durations    = np.diff(events)                   # length N
        descriptions = np.array(new_event_labels)        # length N

        new_ann = mne.Annotations(onset=onsets,
                                duration=durations,
                                description=descriptions)
        raw.set_annotations(new_ann)

        # keep only control samples
        X     = X_full[:, ctrl_mask]
        times = times_full[ctrl_mask]

    # ------------------------------------------------------------------
    # Z-score (per-channel or global)
    # ------------------------------------------------------------------
    def global_scale_channel_center(X):
        """
        Centre each channel by its own mean, then apply a single global
        standard-deviation scale factor.

        Parameters
        ----------
        X : np.ndarray, shape (n_channels, n_times)
            Continuous fNIRS (or EEG/EMG) data; rows are channels,
            columns are time samples.

        Returns
        -------
        X_norm : np.ndarray, same shape as *X*
            Normalised data where
            • every channel has zero mean, and  
            • the entire matrix has unit global standard deviation.

        Notes
        -----
        *Channel-wise centring* removes baseline offsets that differ from
        optode to optode.  *Global scaling* (one σ for all channels) avoids
        the pitfall of inflating low-variance “dead” channels, a concern in
        coma-patient recordings where some optodes may capture only noise.

        Implementation
        --------------
        >>> mu  = X.mean(axis=1, keepdims=True)   # per-channel mean
        >>> sig = X.std()                         # global std (all entries)
        >>> X_norm = (X - mu) / sig

        After this transform:
        •  X_norm.mean(axis=1) ≈ 0   for every channel, and
        •  X_norm.std() == 1         (up to numerical precision).
        """
        mu  = X.mean(axis=1, keepdims=True)   # channel-wise means
        sig = X.std()                         # one global std
        return (X - mu) / (sig+1e-6)

    X = global_scale_channel_center(X)

    return X, raw, sfreq, events, times

# -------------------------------------------------------------------------
# Helper: train one VAE and return per-timepoint label stream
# -------------------------------------------------------------------------
def train_single_model(
    X_t, starts, n_times, n_channels,
    *,
    latent_dim,
    beta,
    seed,
    means=1.2,
    logvar=-1.0,
    epochs_num=25,
    verbose=True,
    device=None,
    k_size=(9,9,7,3), 
    out_channels=(16, 16, 32, 64)
):
    """
    Train ONE Mixture-VAE on a fixed set of sliding windows and produce
    a 1-D numpy array `sample_labels` of length `n_times` containing 0/1
    probabilities (and NaN where samples were removed by the buffer).

    Returns
    -------
    sample_labels : np.ndarray shape (n_times,)
    covered_mask  : np.ndarray bool   shape (n_times,)
        True for samples that belong to ≥1 kept window.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- RNG seeding for reproducibility --------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # --- Data loader -----------------------------------------------------
    loader = DataLoader(TensorDataset(X_t), batch_size=64, shuffle=True)

    # --- Model + simple two-component prior ------------------------------
    model = MixtureVAE(n_channels, X_t.shape[2], latent_dim, k_size=k_size, out_channels=out_channels).to(device)

    prior_means = torch.ones((2, latent_dim), device=device) \
                  * means
    prior_means[0] *= -1  # mirror the second component

    prior_logvars = torch.ones_like(prior_means) \
                    * logvar
    pi_mix = torch.tensor([0.5, 0.5], device=device)

    # --- Train -----------------------------------------------------------
    train_mixture_vae(
        model, loader, loader,                  # same data for validation
        prior_means, prior_logvars, pi_mix,
        epochs_num, device,
        beta=beta,
        verbose=verbose,
        ramp_up=True,
    )

    # --- Inference: nearest-prior classifier per window ------------------
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t.to(device))
        latent = mu.cpu().numpy()

    d0 = ((latent - prior_means[0].cpu().numpy())**2).sum(1)
    d1 = ((latent - prior_means[1].cpu().numpy())**2).sum(1)
    window_labels = np.where(d0 < d1, 0, 1)

    # --- Map window labels → per-sample labels ---------------------------
    sample_labels, covered = per_timepoint_labels_sparse(
        window_labels=window_labels,
        window_length=X_t.shape[2],
        starts=starts,
        n_times=n_times
    )

    # --- Clean up GPU memory --------------------------------------------
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return sample_labels, covered, latent


# -------------------------------------------------------------------------
# helper: run one committee on a *pre-masked* data set
# -------------------------------------------------------------------------

def _run_committee(
        X, times, events, event_names, raw_haemo,
        *, seeds, beta, means, logvar, window_length,
        window_buffer, latent_dim, epochs_num,
        device, k_size, out_channels, verbose, debug):

    n_ch, n_times = X.shape

    # ---------------------------------------------
    # 1) create sliding windows on the *subset*
    # ---------------------------------------------
    X_win, starts = create_sliding_windows_no_classes(
        data=X,
        window_length=window_length,
        times=times,
        events=events,
        buffer=window_buffer
    )
    X_t = torch.from_numpy(X_win)

    if debug:
        # ---------- DEBUG: show per-epoch window counts ----------
        print(f"[DEBUG _run_committee] windows kept: {len(starts)}")
        for i, (s, e, lbl) in enumerate(zip(events[:-1], events[1:], event_names)):
            # “starts” are sample-indices → convert to time-stamp
            w = np.sum((times[starts] >= s) & (times[starts] + window_length / (len(times) / (times[-1]-times[0])) <= e))
            print(f"  subset-epoch {i:2d} [{s:7.1f}→{e:7.1f}s] "
                f"({lbl:7s}): duration {(e-s):5.1f}s, windows={w}")
        print()

    # ---------------------------------------------
    # 2) train every seed
    # ---------------------------------------------
    label_streams, latent_list, covered_mask = [], [], None
    for seed in seeds:
        stream, covered, latent = train_single_model(
            X_t, starts, n_times, n_ch,
            latent_dim=latent_dim, beta=beta,
            means=means, logvar=logvar, seed=seed,
            epochs_num=epochs_num, verbose=verbose,
            device=device, k_size=k_size, out_channels=out_channels,
        )
        label_streams.append(stream)
        latent_list.append(latent)
        covered_mask = covered         # identical for all seeds

    label_streams = np.vstack(label_streams)[:, covered_mask]
    if debug:
        print(f"[DEBUG _run_committee] samples covered: {covered_mask.sum()}\n")

    # ---------------------------------------------
    # 3) align + average
    # ---------------------------------------------
    ref, valid = label_streams[0], ~np.isnan(label_streams[0])
    for i in range(1, label_streams.shape[0]):
        if np.corrcoef(ref[valid], label_streams[i, valid])[0, 1] < 0:
            label_streams[i] = 1.0 - label_streams[i]

    consensus  = np.nanmean(label_streams, axis=0)
    times_trim = times[covered_mask]

    return dict(
        consensus      = consensus,
        times          = times_trim,
        events         = events,
        event_names    = event_names,   # <-- now present
        starts         = starts,            # ← new!
        raw            = raw_haemo,
        latents        = latent_list,
        streams        = label_streams,
    )

# ─────────────────────────────────────────────────────────────
# main entry: build two datasets, run two committees
# ─────────────────────────────────────────────────────────────
# helper ---------------------------------------------------------------
def _get_event_names(raw, events):
    """Return description string for every onset in *events*."""
    onsets = raw.annotations.onset
    descs  = np.asarray(raw.annotations.description)
    names  = []
    for ev in events:
        idx = np.where(np.isclose(onsets, ev))[0]
        names.append(descs[idx[0]] if idx.size else "Unknown")
    return names


# ───────────────────────────────── committee_for_subject ──────────────
def committee_for_subject(
        participant_idx,
        *,
        seeds=(41, 42, 43),
        beta=0.2,
        means=1.0,
        logvar=-1.0,
        window_length=32,
        window_buffer=1.0,
        latent_dim=10,
        epochs_num=25,
        ALL_CONTROL=False,
        verbose=False,
        device=None,
        k_size=(15, 13, 9, 7, 3),
        out_channels=(16, 16, 32, 64, 32),
        debug=False):

    if device is None:
        device = torch.device("cuda"
                              if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ control-only
    if ALL_CONTROL:
        X_ctrl, raw, sfreq, events, times = load_and_preprocess(
            participant_idx, all_control=True)

        res_ctrl = _run_committee(
            X_ctrl, times, events, raw_haemo=raw,
            event_names=_get_event_names(raw, events),
            seeds=seeds, beta=beta, means=means, logvar=logvar,
            window_length=window_length, window_buffer=window_buffer,
            latent_dim=latent_dim, epochs_num=epochs_num,
            device=device, k_size=k_size, out_channels=out_channels,
            verbose=verbose, debug=debug)

        return {"all_control": res_ctrl}

    # ------------------------------------------------------------------ full recording
    X_full, raw, sfreq, events_full, times_full = load_and_preprocess(
        participant_idx, all_control=False)

    # ----- build (cond, start, end) tuples ----------------------------
    labels = []
    for i, desc in enumerate(raw.annotations.description):
        start = raw.annotations.onset[i]
        end   = (raw.annotations.onset[i+1] if i+1 < len(raw.annotations.onset)
                 else times_full[-1] + 1/sfreq)
        cond = ("Right"  if "Tapping_Right" in desc else
                "Left"   if "Tapping_Left"  in desc else
                "Control")
        labels.append((cond, start, end))

    def subset(keep):
        mask   = np.zeros_like(times_full, dtype=bool)
        ev_out = []; names_out = []
        for cond, s, e in labels:
            if cond in keep:
                mask |= (times_full >= s) & (times_full < e)
                ev_out.append(s);  names_out.append(cond)
        ev_out.append(times_full[-1] + 1/sfreq)
        return (X_full[:, mask],
                times_full[mask],
                np.array(ev_out),
                names_out)                      # list of labels

    # ---------- Right vs Control ----------
    X_r, t_r, ev_r, names_r = subset({"Control", "Right"})
    res_right = _run_committee(
        X_r, t_r, ev_r, raw_haemo=raw,
        event_names=names_r,
        seeds=seeds, beta=beta, means=means, logvar=logvar,
        window_length=window_length, window_buffer=window_buffer,
        latent_dim=latent_dim, epochs_num=epochs_num,
        device=device, k_size=k_size, out_channels=out_channels,
        verbose=verbose, debug=debug)

    # ---------- Left vs Control ----------
    X_l, t_l, ev_l, names_l = subset({"Control", "Left"})
    res_left = _run_committee(
        X_l, t_l, ev_l, raw_haemo=raw,
        event_names=names_l,
        seeds=seeds, beta=beta, means=means, logvar=logvar,
        window_length=window_length, window_buffer=window_buffer,
        latent_dim=latent_dim, epochs_num=epochs_num,
        device=device, k_size=k_size, out_channels=out_channels,
        verbose=verbose, debug=debug)

    return {"right": res_right, "left": res_left}


# -------------------------------------------------------------------------
# Simple CLI test run
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # example usage:
    latent_dim=2
    window_buffer = 1
    consensus, times_trimmed, events, raw_haemo, latent_list, label_streams = committee_for_subject(
        participant_idx = 4,
        seeds           = (1,),
        means           = 1.5 ,
        logvar          = -0.3,
        beta            = 0.35,
        window_length   = 48  ,
        window_buffer   = 2.3 ,
        latent_dim      = 8   ,
        epochs_num      = 200 ,
        verbose         = False,
        ALL_CONTROL     = True
    )

    # -----------------------------------------------------------------
    # 1) Basic renaming + event labels from annotations
    # -----------------------------------------------------------------
    labels_trimmed = consensus        # float in [0,1]
    times_trimmed  = times_trimmed           # seconds (same length)
    events         = np.sort(events)         # ensure ascending

    onsets = raw_haemo.annotations.onset
    descs  = np.asarray(raw_haemo.annotations.description)

    event_names = []
    for ev in events:
        idx = np.where(np.isclose(onsets, ev))[0]
        event_names.append(descs[idx[0]] if idx.size else "Unknown")

    # -----------------------------------------------------------------
    # 2) Bucket every individual sample by its interval type
    # -----------------------------------------------------------------

    samples_control, samples_tap_left, samples_tap_right = [], [], []

    for (start, end), name in zip(zip(events[:-1], events[1:]), event_names[:-1]):
        mask = (times_trimmed >= start) & (times_trimmed < end)
        if not mask.any():
            continue
        vals = labels_trimmed[mask]
        if "Control" in name:
            samples_control.extend(vals)
        elif "Tapping_Left" in name or "Tapping/Left" in name:
            samples_tap_left.extend(vals)
        elif "Tapping_Right" in name or "Tapping/Right" in name:
            samples_tap_right.extend(vals)

    # -----------------------------------------------------------------
    # 3) Piece-wise constant trace y(t) = interval mean
    # -----------------------------------------------------------------
    y = np.full_like(labels_trimmed, np.nan, dtype=float)
    interval_means = []

    for start, end in zip(events[:-1], events[1:]):
        mask = (times_trimmed >= start) & (times_trimmed < end)
        if mask.any():
            m = labels_trimmed[mask].mean()
            y[mask] = m
            interval_means.append(m)

    plt.figure(figsize=(10, 4))
    plt.plot(times_trimmed, y, lw=2)
    for ev in events:
        plt.axvline(ev, color='gray', ls='--', alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Committee avg.\n(predicted label per interval)")
    plt.title("Event-interval averaged committee prediction")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # 4) Histograms by bucket + simple stats
    # -----------------------------------------------------------------
    print("Means ± SEM")
    for title, arr in [("Control", samples_control),
                    ("Tapping-Left", samples_tap_left),
                    ("Tapping-Right", samples_tap_right)]:
        if arr:
            mu  = np.mean(arr)
            sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
            print(f"{title:14s}: {mu:.3f} ± {sem:.3f}")
        else:
            print(f"{title:14s}: (no data)")

    buckets = [
        ("Control",       samples_control),
        ("Tapping-Left",  samples_tap_left),
        ("Tapping-Right", samples_tap_right),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for ax, (title, vals) in zip(axes, buckets):
        if vals:
            ax.hist(vals, bins='auto', edgecolor='k', alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No data",
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        if ax is axes[0]:
            ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()