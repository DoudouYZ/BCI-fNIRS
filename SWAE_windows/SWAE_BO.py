#!/usr/bin/env python3
"""
Bayesian optimisation of SWAE hyper-parameters.

Goal
-----
Find a single hyper-parameter set that

  • minimises the average p-value obtained when
    CONTROL_SPLIT == False  (tested twice: USE_LEFT_HAND = True & False)
  • maximises the p-value obtained when
    CONTROL_SPLIT == True   (two control halves should look alike)

The objective we *minimise* is
    score = p_left + p_right + (1.0 - p_split)

Smaller is better ⇒  ideal is 0 (left & right highly significant, split ~1).

Notes
-----
* Requires Optuna ≥ 3.0 (`pip install optuna`) and the same project-level
  modules your original training script imports.
* Uses one GPU (if available) but frees CUDA memory between trials.
* Each trial loops over all subjects so results are immediately comparable.
* Adjust N_TRIALS to taste; 30–50 is usually enough to see convergence.

Author: ChatGPT (o3)
Date  : 2025-06-15
"""

import argparse, gc, sys, time
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
from sklearn.exceptions import ConvergenceWarning

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.stats import fisher_exact
import optuna                                   # Bayesian optimiser
import mne                                      # used inside training
from torch.utils.data import DataLoader

# ── project-local imports (unchanged from your code) ──────────────────────────
from SWAE_preprocessing import preprocess_for_wae_windows
from SWAE_model import (WindowDataset, build_dataloader,
                        Encoder2D, Decoder2D,
                        GMMPrior, sliced_wasserstein)
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUBJECTS = (0,1,2,3)
ALPHA = 0.3          # weight of the control term; tweak 0.1–0.3 to taste
P_THRESH = 0.05      # significance threshold
LEARNABLE_PRIOR = False
WINDOW_SIZE     = 116
WINDOW_STEP     = 32
batch_size      = 64
EPOCHS          = 250
Z_DIM           = 11
LR              = 1e-3
LAMBDA_SWD      = 5
WARMUP_EPOCHS   = 100
N_PROJ          = 465
PRIOR_MEAN      = 0.5
PRIOR_LOGVAR    = 0.5

def get_or_create_study(args) -> optuna.Study:
    """
    Returns an Optuna study backed by a SQLite file that survives script restarts.
    If the file exists the previous trials are loaded automatically.
    """
    study_path = Path(__file__).with_suffix(".db")      # e.g. swae_bayes_optimize.db
    storage    = f"sqlite:///{study_path}"
    return optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True
    )

def _clean_latents(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaNs/±Infs produced by divide-by-zero normalisation with zeros.
    Keeps Optuna running while still penalising bad hyper-params.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def safe_build_dataloader(matrices, batch, shuffle=True, drop_last=False):
    """
    Wrapper around the original build_dataloader that guarantees
    at least one batch (no empty DataLoader → no np.concatenate error).
    The signature matches the original, so callers can still pass drop_last.
    """
    ds = WindowDataset(matrices)
    batch = min(batch, len(ds)) or 1            # never zero
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        drop_last=False                         # we always keep the final (small) batch
    )

# monkey-patch global reference so every later call uses the safe version
build_dataloader = safe_build_dataloader


# ═════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═════════════════════════════════════════════════════════════════════════════
def one_training_pass(subj: int,
                      params: Dict,
                      control_split: bool,
                      use_left_hand: bool) -> float:
    """
    Train a SWAE for one subject with the given hyper-parameters and return the
    Fisher exact p-value comparing the two classes present in this setting.
    """
    # --- 1. Load epochs ------------------------------------------------------
    full_epochs = get_raw_subject_data(subj)
    tap_lbl = "Tapping_Left" if use_left_hand else "Tapping_Right"
    ctl_lbl = "Control"

    if control_split:
        # Duplicate & split 60-s control epochs into 2×20-s halves
        sfreq = full_epochs.info['sfreq']
        full_epochs = get_raw_subject_data(subj, tmin=-30, tmax=30)
        control_epochs = full_epochs["Control"]

        halves = []
        for epoch in control_epochs:
            halves.append(epoch[:, :int(20 * sfreq)])                   # first 20 s
            halves.append(epoch[:, int(40 * sfreq): int(60 * sfreq)])   # last 20 s

        idx = np.random.permutation(len(halves))
        a_idx, b_idx = idx[:len(idx)//2], idx[len(idx)//2:]
        info = full_epochs["Control"].info.copy()
        epochs_dict = {
            "Control_A": mne.EpochsArray(np.stack([halves[i] for i in a_idx]), info, tmin=-30),
            "Control_B": mne.EpochsArray(np.stack([halves[i] for i in b_idx]), info, tmin=-30),
        }
        class_labels = ["Control_A", "Control_B"]

    else:
        epochs_dict = {tap_lbl: full_epochs[tap_lbl], ctl_lbl: full_epochs[ctl_lbl]}
        class_labels = [tap_lbl, ctl_lbl]

    # --- 2. Convert to time-window matrices ----------------------------------
    mats = preprocess_for_wae_windows(
        epochs_dict,
        class_labels=class_labels,
        window_size=params["WINDOW_SIZE"],
        window_step=params["WINDOW_STEP"],
    )
    BATCH_SIZE = 64
    if BATCH_SIZE > len(mats[class_labels[0]]):
        BATCH_SIZE = len(mats[class_labels[0]])
    X  = mats[class_labels[0]] + mats[class_labels[1]]
    y  = np.array([0]*len(mats[class_labels[0]]) + [1]*len(mats[class_labels[1]]))

    # --- 3. Build model ------------------------------------------------------
    ds  = WindowDataset(X);   C, T = ds.C, ds.T
    enc = Encoder2D(C, params["Z_DIM"]).to(DEVICE)
    dec = Decoder2D(C, T, params["Z_DIM"]).to(DEVICE)
    pri = GMMPrior(
        params["Z_DIM"],
        mu=params["PRIOR_MEAN"],
        logvar=params["PRIOR_LOGVAR"],
        learnable=LEARNABLE_PRIOR
    ).to(DEVICE)

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()) + list(pri.parameters()),
        lr=LR
    )
    mse = torch.nn.MSELoss()
    dl  = build_dataloader(X, BATCH_SIZE, shuffle=True, drop_last=True)

    # --- 4. Train ------------------------------------------------------------
    for ep in range(1, params["EPOCHS"] + 1):
        if ep < params["WARMUP_EPOCHS"] * 0.25:
            current_lambda = 0.0
        elif ep < params["WARMUP_EPOCHS"]:
            frac = (ep - params["WARMUP_EPOCHS"] * 0.25) / (params["WARMUP_EPOCHS"] * 0.75)
            current_lambda = params["LAMBDA_SWD"] * frac
        else:
            current_lambda = params["LAMBDA_SWD"]

        for xb in dl:
            xb = xb.to(DEVICE)
            mu, logv = enc(xb)
            z  = mu + torch.exp(0.5 * logv) * torch.randn_like(mu)
            xb_hat = dec(z)
            loss = mse(xb_hat, xb) + current_lambda * sliced_wasserstein(
                z, pri.sample(xb.size(0)), n_proj=params["N_PROJ"]
            )
            opt.zero_grad(); loss.backward(); opt.step()

    # --- 5. Extract features & evaluate --------------------------------------
    with torch.no_grad():
        zs = [enc(xb.to(DEVICE))[0].cpu().numpy() for xb in dl]
        if not zs:                       # should never happen after safe DataLoader
            return 1.0                   # worst p-value → big penalty
        Z = _clean_latents(np.concatenate(zs))

    # guard against singular matrix after cleaning
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf = LogisticRegression(max_iter=1000)
            preds = clf.fit(Z, y[:len(Z)]).predict(Z)
    except Exception:                    # any numerical failure
        return 1.0                       # penalise but do not crash the trial

    tn, fp, fn, tp = confusion_matrix(y[:len(Z)], preds, labels=[0, 1]).ravel()
    table = [[tn, fp], [fn, tp]]
    return fisher_exact(table)[1]

def aggregate_p_values(params: Dict) -> Tuple[float, float, float]:
    """
    Compute mean p-values across subjects for the three evaluation conditions.
    Returns (p_left, p_right, p_split).
    """
    p_left  = np.mean([
        one_training_pass(s, params, control_split=False, use_left_hand=True)
        for s in SUBJECTS
    ])
    p_right = np.mean([
        one_training_pass(s, params, control_split=False, use_left_hand=False)
        for s in SUBJECTS
    ])
    p_split = np.mean([
        one_training_pass(s, params, control_split=True,  use_left_hand=True)
        for s in SUBJECTS
    ])
    return p_left, p_right, p_split


# ═════════════════════════════════════════════════════════════════════════════
#  Optuna objective
# ═════════════════════════════════════════════════════════════════════════════
def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective emphasising the 0.05 threshold for tapping tests."""
    params = {
        "WINDOW_SIZE":     trial.suggest_int("WINDOW_SIZE", 30, 120),
        "WINDOW_STEP":     trial.suggest_int("WINDOW_STEP", 1, 70),
        "LEARNABLE_PRIOR": trial.suggest_categorical("LEARNABLE_PRIOR", [False, True]),
        "PRIOR_MEAN":      trial.suggest_float("PRIOR_MEAN", -2.0, 0.5),
        "PRIOR_LOGVAR":    trial.suggest_float("PRIOR_LOGVAR", -2.0, -0.2),
        "EPOCHS":          trial.suggest_int("EPOCHS", 70, 200),
        "Z_DIM":           trial.suggest_int("Z_DIM", 6, 16),
        "LAMBDA_SWD":      trial.suggest_float("LAMBDA_SWD", 0.1, 4.0, log=True),
        "WARMUP_EPOCHS":   trial.suggest_int("WARMUP_EPOCHS", 6, 30),
        "N_PROJ":          trial.suggest_int("N_PROJ", 50, 800),
    }
    # ws = trial.suggest_int("WINDOW_SIZE", 30, 120)
    params = {
        "WINDOW_SIZE":     WINDOW_SIZE,
        "WINDOW_STEP":     WINDOW_STEP,
        "PRIOR_MEAN":      PRIOR_MEAN,
        "PRIOR_LOGVAR":    trial.suggest_float("PRIOR_LOGVAR", -2.0, 2.0),
        "Z_DIM":           trial.suggest_int("Z_DIM", 7, 20),
        "LAMBDA_SWD":      trial.suggest_float("LAMBDA_SWD", 0.7, 10, log=True),
        "EPOCHS":          EPOCHS,
        "WARMUP_EPOCHS":   trial.suggest_int("WARMUP_EPOCHS", 50, 200),
        "N_PROJ":          N_PROJ,
    }
    if params["WARMUP_EPOCHS"] > params["EPOCHS"]:
        return float("inf")          # skip impossible combos

    p_left, p_right, p_split = aggregate_p_values(params)

    tap_pen = ((p_left - P_THRESH) if p_left >= P_THRESH else 0.3 * (P_THRESH - p_left)) + \
              ((p_right - P_THRESH) if p_right >= P_THRESH else 0.3 * (P_THRESH - p_right))
    if p_split < P_THRESH:
        ctrl_pen = 5.0 * (1.0 - p_split)  # Apply heavy penalty when p_split is low
    elif p_split > 0.2:
        ctrl_pen = 0.0
    else:
        ctrl_pen = 1.0 - p_split


    loss = tap_pen + ALPHA * ctrl_pen

    trial.set_user_attr("p_left", p_left)
    trial.set_user_attr("p_right", p_right)
    trial.set_user_attr("p_split", p_split)
    trial.set_user_attr("loss_components",
                        {"tap_pen": tap_pen, "ctrl_pen": ctrl_pen})

    return loss


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Bayesian optimisation for SWAE.")
    ap.add_argument("--trials", type=int, default=150, help="number of Optuna trials")
    ap.add_argument("--study-name", default="swae_bayes_opt_last10", help="Optuna study name") #swae_bayes_opt_mega has 300+ trials, also ctrlpenal
    args = ap.parse_args()

    study = get_or_create_study(args)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    print("\nBest hyper-parameters (score = {:.4f}):".format(best.value))
    for k, v in best.params.items():
        print(f"  {k:<15} : {v}")
    print("Mean p-values for that trial:")
    print("  CONTROL_SPLIT = False | left hand : {:.4g}".format(best.user_attrs["p_left"]))
    print("  CONTROL_SPLIT = False | right hand: {:.4g}".format(best.user_attrs["p_right"]))
    print("  CONTROL_SPLIT = True              : {:.4g}".format(best.user_attrs["p_split"]))

    # Optional: save study for later inspection
    study_path = Path(__file__).with_suffix(".db")
    study.storage = f"sqlite:///{study_path}"     # saves automatically
    print(f"\nStudy saved to {study_path}")


if __name__ == "__main__":
    main()
