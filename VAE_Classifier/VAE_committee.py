"""
vae_committee.py – Committee trainer upgraded for optional WAE‑MMD.
Only the **signatures** and the call into train_mixture_vae changed.
"""

from __future__ import annotations
import gc, random, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mne

# ------------------------------------------------------------------
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from AE_models import *
from Preprocessing import get_continuous_subject_data, get_raw_subject_data

# ───────────────────────────── helper: event-name lookup ──────────
def _get_event_names(raw: mne.io.BaseRaw, events: np.ndarray) -> list[str]:
    """Return list of annotation descriptions aligned with every *events* onset."""
    on, desc = raw.annotations.onset, np.asarray(raw.annotations.description)
    out = []
    for ev in events:
        idx = np.where(np.isclose(on, ev))[0]
        out.append(desc[idx[0]] if idx.size else "Unknown")
    return out


# ───────────────────────────── load + basic preprocess ─────────────
def load_and_preprocess(
        participant_idx: int,
        *,
        all_control: bool = False,
        fake_cycle=("Control", "Tapping_Left", "Tapping_Right")):
    """
    • If *all_control* is False  → ordinary load (keep every epoch)
    • If *all_control* is True   → keep only control epochs and relabel them
                                   cyclically Control → TL → TR → …
    Returns
    -------
    X, raw, sfreq, events, times
    """
    raw, sfreq, events_orig = get_continuous_subject_data(subject=participant_idx)
    X_full   = raw.get_data()
    times_full = raw.times

    if not all_control:
        X, times, events = X_full, times_full, events_orig

    else:                                       # -------- control-only mode
        on, desc = raw.annotations.onset, raw.annotations.description
        ctrl_mask = np.zeros(times_full.size, bool)
        new_ev_starts, new_labels = [], []

        cycle_len, ctrl_count = len(fake_cycle), 0
        for i, (start, d) in enumerate(zip(on, desc)):
            end = on[i+1] if i+1 < len(on) else times_full[-1] + 1/sfreq
            if "Control" not in d:
                continue
            ctrl_mask |= (times_full >= start) & (times_full < end)
            new_ev_starts.append(start)
            new_labels.append(fake_cycle[ctrl_count % cycle_len])
            ctrl_count += 1

        events = np.array(new_ev_starts + [times_full[-1] + 1/sfreq])
        raw.set_annotations(
            mne.Annotations(onset=np.asarray(new_ev_starts),
                            duration=np.diff(events),
                            description=np.asarray(new_labels)))

        X, times = X_full[:, ctrl_mask], times_full[ctrl_mask]

    # -------- global z-score (mean 0 per-channel, single global σ) ----
    mu = X.mean(axis=1, keepdims=True)
    sig = X.std()
    X = (X - mu) / (sig + 1e-6)

    return X, raw, sfreq, events, times


# ───────────────────────────── train one VAE model ─────────────────
# ─────────────────────────── train one model ───────────────────────────

def train_single_model(X_t: torch.Tensor, starts: np.ndarray, epoch_ids: np.ndarray,
                       n_times: int, n_channels: int,
                       *, latent_dim: int = 8,
                       beta: float = 0.2,
                       use_mmd: bool = True,
                       lam_mmd: float = 10.0,
                       mmd_sigma: float = 1.0,
                       seed: int = 42,
                       means: float = 1.2,
                       logvar: float = -1.0,
                       epochs_num: int = 25,
                       device=None,
                       k_size=(9, 9, 7, 7, 3),
                       out_channels=(16, 16, 32, 64, 32),
                       verbose=False):
    """Train one MixtureVAE (or WAE‑MMD) with epoch information and return the label stream."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import random, gc, torch, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    
    # Build a dataset from the windows and the computed epoch_ids.
    from torch.utils.data import DataLoader, TensorDataset
    # Convert epoch_ids to tensor
    epoch_ids_tensor = torch.from_numpy(epoch_ids)
    dataset = TensorDataset(X_t, epoch_ids_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # model uses NewMixtureVAE from AE_models.py
    from AE_models import train_mixture_vae  # use the modified trainer
    model = NewMixtureVAE(X_t.shape[2], latent_dim, k_size=k_size, out_channels=out_channels).to(device)

    prior_mu  = torch.full((2, latent_dim), means, device=device)
    prior_mu[0] *= -1
    prior_lv  = torch.full_like(prior_mu, logvar)
    pi_mix    = torch.tensor([0.5, 0.5], device=device)

    # Split loader into training and validation; here we use 90/10 split.
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    from torch.utils.data import random_split
    train_ds, val_ds = random_split(dataset, [n_train, n_total - n_train])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Use an added consistency_weight to nudge windows from the same epoch to agree.
    # consistency_weight = 1.0  # adjust weight as needed

    hist = train_mixture_vae(model, train_loader, val_loader,
                             prior_mu, prior_lv, pi_mix,
                             epochs_num, device,
                             beta=beta,
                             use_mmd=use_mmd,
                             lam_mmd=lam_mmd,
                             mmd_sigma=mmd_sigma,
                             verbose=verbose)

    # ----- get latents + assign hard labels -----
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(X_t.to(device))
    latent = mu.cpu().numpy()
    d0 = ((latent - prior_mu[0].cpu().numpy()) ** 2).sum(1)
    d1 = ((latent - prior_mu[1].cpu().numpy()) ** 2).sum(1)
    win_labels = np.where(d0 < d1, 0, 1)

    sample_labels, covered = per_timepoint_labels_sparse(
        win_labels, X_t.shape[2], starts, n_times)

    del model; torch.cuda.empty_cache(); gc.collect()
    return sample_labels, covered, latent


# ───────────────────────────── run committee on subset ────────────
# def _run_committee(X, times, events, event_names, raw,
#                    *, seeds, beta, means, logvar,
#                    window_length, window_buffer,
#                    latent_dim, epochs_num,
#                    device, k_size, out_channels,
#                    use_mmd=True, lam_mmd=10.0, mmd_sigma=1.0,
#                    verbose, debug=False):

#     n_ch, n_times = X.shape
#     X_win, starts = create_sliding_windows_no_classes(
#         X, window_length, times, events, buffer=window_buffer)
#     X_t = torch.from_numpy(X_win)

#     if debug:
#         print(f"[DEBUG] windows kept   : {len(starts)}")

#     label_streams, latent_list, covered = [], [], None
#     for sd in seeds:
#         stream, cvd, latent = train_single_model( X_t, starts, n_times, n_ch,
#                                                   latent_dim=latent_dim, beta=beta,
#                                                   use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
#                                                   means=means, logvar=logvar, seed=sd,
#                                                   epochs_num=epochs_num, device=device,
#                                                   k_size=k_size, out_channels=out_channels,
#                                                   verbose=verbose)
         
#         label_streams.append(stream); latent_list.append(latent); covered=cvd

#     label_streams = np.vstack(label_streams)[:, covered]
#     ref, valid = label_streams[0], ~np.isnan(label_streams[0])
#     for i in range(1, label_streams.shape[0]):
#         if np.corrcoef(ref[valid], label_streams[i, valid])[0,1] < 0:
#             label_streams[i] = 1 - label_streams[i]
#     consensus = np.nanmean(label_streams, axis=0)
#     times_trim = times[covered]

#     if debug:
#         print(f"[DEBUG] samples covered: {covered.sum()}")

#     return dict(consensus=consensus,
#                 times=times_trim,
#                 events=events,
#                 event_names=event_names,
#                 starts=starts,
#                 raw=raw,
#                 latents=latent_list,
#                 streams=label_streams)

def _run_committee(X, times, events, event_names, raw,
                   *, seeds, beta, means, logvar,
                   window_length, window_buffer,
                   latent_dim, epochs_num,
                   device, k_size, out_channels,
                   use_mmd=True, lam_mmd=10.0, mmd_sigma=1.0,
                   verbose, debug=False):

    n_ch, n_times = X.shape
    # Now also get epoch_ids from the windowing helper.
    X_win, starts, epoch_ids = create_sliding_windows_no_classes(
        X, window_length, times, events, buffer=window_buffer)
    X_t = torch.from_numpy(X_win)

    if debug:
        print(f"[DEBUG] windows kept   : {len(starts)}")

    label_streams, latent_list, covered = [], [], None
    for sd in seeds:
        stream, cvd, latent = train_single_model( X_t, starts, epoch_ids,
                                                  n_times, n_ch,
                                                  latent_dim=latent_dim, beta=beta,
                                                  use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
                                                  means=means, logvar=logvar, seed=sd,
                                                  epochs_num=epochs_num, device=device,
                                                  k_size=k_size, out_channels=out_channels,
                                                  verbose=verbose)
        label_streams.append(stream)
        latent_list.append(latent)
        covered = cvd

    label_streams = np.vstack(label_streams)[:, covered]
    ref, valid = label_streams[0], ~np.isnan(label_streams[0])
    for i in range(1, label_streams.shape[0]):
        if np.corrcoef(ref[valid], label_streams[i, valid])[0, 1] < 0:
            label_streams[i] = 1 - label_streams[i]
    consensus = np.nanmean(label_streams, axis=0)
    times_trim = times[covered]

    if debug:
        print(f"[DEBUG] samples covered: {covered.sum()}")

    return dict(consensus=consensus,
                times=times_trim,
                events=events,
                event_names=event_names,
                starts=starts,
                raw=raw,
                latents=latent_list,
                streams=label_streams)


# ───────────────────────────── public entry point ─────────────────
# def committee_for_subject(
#         participant_idx: int,
#         *,
#         seeds=(41,42,43),
#         beta=0.2, means=1.0, logvar=-1.0,
#         window_length=32, window_buffer=1.0,
#         latent_dim=10, epochs_num=25,
#         use_mmd=True, lam_mmd=10.0, mmd_sigma=1.0,
#         ALL_CONTROL=False, verbose=False, debug=False,
#         device=None,
#         k_size=(15,13,9,7,3),
#         out_channels=(16,16,32,64,32)):
#     """
#     Returns
#     -------
#     { "right": {...}, "left": {...} }           (normal run)
#     { "all_control": {...} }                    (if ALL_CONTROL=True)
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ------------------------- control-only quick-path ------------------
#     if ALL_CONTROL:
#         Xc, raw, sfreq, evc, tc = load_and_preprocess(participant_idx, all_control=True)
#         res_ctrl = _run_committee(
#             Xc, tc, evc, _get_event_names(raw, evc), raw,
#             use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
#             seeds=seeds, beta=beta, means=means, logvar=logvar,
#             window_length=window_length, window_buffer=window_buffer,
#             latent_dim=latent_dim, epochs_num=epochs_num,
#             device=device, k_size=k_size, out_channels=out_channels,
#             verbose=verbose, debug=debug)
#         return {"all_control": res_ctrl}

#     # ------------------------- full recording once ---------------------
#     X_full, raw, sfreq, events_full, times_full = load_and_preprocess(
#         participant_idx, all_control=False)

#     on, desc = raw.annotations.onset, raw.annotations.description
#     epochs = [(("Right"  if "Tapping_Right" in d else
#                 "Left"   if "Tapping_Left"  in d else
#                 "Control"),
#                on[i], on[i+1] if i+1<len(on) else times_full[-1]+1/sfreq)
#               for i,d in enumerate(desc)]

#     def subset(cond_keep):
#         mask = np.zeros_like(times_full, bool)
#         ev, names = [], []
#         for cond,s,e in epochs:
#             if cond in cond_keep:
#                 mask |= (times_full>=s)&(times_full<e)
#                 ev.append(s); names.append(cond)
#         ev.append(times_full[-1] + 1/sfreq)
#         return X_full[:,mask], times_full[mask], np.array(ev), names

#     # -------------- RIGHT committee --------------
#     X_r,t_r,ev_r,nm_r = subset({"Control","Right"})
#     res_r = _run_committee(
#         X_r,t_r,ev_r,nm_r,raw,
#         use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
#         seeds=seeds,beta=beta,means=means,logvar=logvar,
#         window_length=window_length,window_buffer=window_buffer,
#         latent_dim=latent_dim,epochs_num=epochs_num,
#         device=device,k_size=k_size,out_channels=out_channels,
#         verbose=verbose,debug=debug)

#     # -------------- LEFT committee ---------------
#     X_l,t_l,ev_l,nm_l = subset({"Control","Left"})
#     res_l = _run_committee(
#         X_l,t_l,ev_l,nm_l,raw,
#         use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
#         seeds=seeds,beta=beta,means=means,logvar=logvar,
#         window_length=window_length,window_buffer=window_buffer,
#         latent_dim=latent_dim,epochs_num=epochs_num,
#         device=device,k_size=k_size,out_channels=out_channels,
#         verbose=verbose,debug=debug)

#     return {"right": res_r, "left": res_l}


def committee_for_subject(
        participant_idx: int,
        *,
        seeds=(41,42,43),
        beta=0.2, means=1.0, logvar=-1.0,
        window_length=32, window_buffer=1.0,
        latent_dim=10, epochs_num=25,
        use_mmd=True, lam_mmd=10.0, mmd_sigma=1.0,
        ALL_CONTROL=False, use_epochs=False,         # << added use_epochs
        verbose=False, debug=False,
        device=None,
        k_size=(15,13,9,7,3),
        out_channels=(16,16,32,64,32)):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # —————— EPOCHS‐BASED QUICK PATH ——————
    if use_epochs:
        # (1) load MNE epochs, get numpy array
        epochs = get_raw_subject_data(subject=participant_idx)
        sfreq  = epochs.info["sfreq"]
        data   = epochs.get_data()                     # (n_epochs, n_ch, T)
        n_epochs, n_ch, T = data.shape

        # (2) concatenate epochs along time
        X_concat  = data.transpose(1,0,2).reshape(n_ch, n_epochs * T)
        times      = np.arange(n_epochs * T) / sfreq    # dummy time axis in seconds
        events     = np.arange(0, (n_epochs+1)*T, T) / sfreq
        event_names = ["Epoch"] * len(events)

        # (3) hand off to your existing runner
        res_epochs = _run_committee(
            X_concat, times, events, event_names, epochs,
            use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
            seeds=seeds, beta=beta, means=means, logvar=logvar,
            window_length=window_length, window_buffer=window_buffer,
            latent_dim=latent_dim, epochs_num=epochs_num,
            device=device, k_size=k_size, out_channels=out_channels,
            verbose=verbose, debug=debug
        )
        return {"epochs": res_epochs}

    # ------------------------- control-only quick-path ------------------
    if ALL_CONTROL:
        Xc, raw, sfreq, evc, tc = load_and_preprocess(participant_idx, all_control=True)
        res_ctrl = _run_committee(
            Xc, tc, evc, _get_event_names(raw, evc), raw,
            use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
            seeds=seeds, beta=beta, means=means, logvar=logvar,
            window_length=window_length, window_buffer=window_buffer,
            latent_dim=latent_dim, epochs_num=epochs_num,
            device=device, k_size=k_size, out_channels=out_channels,
            verbose=verbose, debug=debug)
        return {"all_control": res_ctrl}

    # ------------------------- full recording once ---------------------
    X_full, raw, sfreq, events_full, times_full = load_and_preprocess(
        participant_idx, all_control=False)

    on, desc = raw.annotations.onset, raw.annotations.description
    epochs = [(("Right"  if "Tapping_Right" in d else
                "Left"   if "Tapping_Left"  in d else
                "Control"),
               on[i], on[i+1] if i+1<len(on) else times_full[-1]+1/sfreq)
              for i,d in enumerate(desc)]

    def subset(cond_keep):
        mask = np.zeros_like(times_full, bool)
        ev, names = [], []
        for cond,s,e in epochs:
            if cond in cond_keep:
                mask |= (times_full>=s)&(times_full<e)
                ev.append(s); names.append(cond)
        ev.append(times_full[-1] + 1/sfreq)
        return X_full[:,mask], times_full[mask], np.array(ev), names

    # -------------- RIGHT committee --------------
    X_r,t_r,ev_r,nm_r = subset({"Control","Right"})
    res_r = _run_committee(
        X_r,t_r,ev_r,nm_r,raw,
        use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
        seeds=seeds,beta=beta,means=means,logvar=logvar,
        window_length=window_length,window_buffer=window_buffer,
        latent_dim=latent_dim,epochs_num=epochs_num,
        device=device,k_size=k_size,out_channels=out_channels,
        verbose=verbose,debug=debug)

    # -------------- LEFT committee ---------------
    X_l,t_l,ev_l,nm_l = subset({"Control","Left"})
    res_l = _run_committee(
        X_l,t_l,ev_l,nm_l,raw,
        use_mmd=use_mmd, lam_mmd=lam_mmd, mmd_sigma=mmd_sigma,
        seeds=seeds,beta=beta,means=means,logvar=logvar,
        window_length=window_length,window_buffer=window_buffer,
        latent_dim=latent_dim,epochs_num=epochs_num,
        device=device,k_size=k_size,out_channels=out_channels,
        verbose=verbose,debug=debug)

    return {"right": res_r, "left": res_l}


# -------------------------- simple CLI test ---------------------------
if __name__ == "__main__":
    out = committee_for_subject(4, seeds=(1,), window_length=48,
                                window_buffer=2.3, latent_dim=8,
                                epochs_num=50, verbose=True, debug=True)
    print("Returned keys:", list(out.keys()))
