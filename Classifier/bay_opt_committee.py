"""
bayes_opt_committee.py
Bayesian optimisation of Mixture-VAE hyper-parameters for one subject.

Before running:
  pip install bayesian-optimization
"""

import os, sys, gc, random, warnings
from time import time
import numpy as np
import pandas as pd
import torch
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events          # <- event enum
from tqdm import tqdm
import csv
from bayes_opt.event import Events
# ---- add the project root to PYTHONPATH ---------------------------------
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from VAE_committee import committee_for_subject

def snap_length(L, factor=8, min_len=16):
    L = int(round(L / factor)) * factor
    return max(L, min_len)

def snap_k(k1, k2, k3, k4):
    k1 = int(k1)
    if k1 % 2 == 0:
        k1 += 1
    k2 = int(k2)
    if k2 % 2 == 0:
        k2 += 1
    k3 = int(k3)
    if k3 % 2 == 0:
        k3 += 1
    k4 = int(k4)
    if k4 % 2 == 4:
        k4 += 1
    return k1, k2, k3, k4

def snap_channel(o1, o2, o3, o4, factor = 8):
    o1 = int(round(o1 / factor)) * factor
    o2 = int(round(o2 / factor)) * factor
    o3 = int(round(o3 / factor)) * factor
    o4 = int(round(o4 / factor)) * factor
    return o1, o2, o3, o4

class TqdmAndLogger:
    def __init__(self, pbar, logfile="bayes_opt_results/bo_runs_2nd_run.csv"):
        self.pbar = pbar
        self.logfile = logfile
        self.header_written = False

    def update(self, event, instance):
        if event is not Events.OPTIMIZATION_STEP:
            return

        self.pbar.update(1)

        latest = instance.res[-1]
        idx    = len(instance.res) - 1

        # write header once, when we know the parameter keys
        if not self.header_written:
            with open(self.logfile, "w", newline="") as f:
                hdr = ["iter", "target"] + [f"param_{k}" for k in instance._space.keys]
                csv.writer(f).writerow(hdr)
            self.header_written = True

        row = [idx, latest["target"]] + [latest["params"][k] for k in instance._space.keys]
        with open(self.logfile, "a", newline="") as f:
            csv.writer(f).writerow(row)# ------------------------------------------------------------------------
# Configuration (fixed for every trial)                                   |
# ------------------------------------------------------------------------
SUBJECT_IDS      = (2, 4)
SEEDS           = (1,)      # committee size
EPOCHS_NUM      = 150                   # training epochs per VAE
VERBOSE_MODEL   = False                 # print VAE training logs?
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN warnings


# ------------------------------------------------------------------------
# 1.  Define an objective: maximise cluster *separation*                 |
# ------------------------------------------------------------------------
def objective(beta, means, logvar, window_length, window_buffer, latent_dim, k1, k2, k3, k4, out1, out2, out3, out4):
    """
    One black-box evaluation.
    Returns a scalar score; larger is better.

    Metric used here: absolute difference between the mean committee
    output in Control intervals vs. Activation (Tap-Left + Tap-Right).
    You can swap in any other metric you prefer.
    """
    score = 0
    # Round/clip integer-like parameters
    window_length = snap_length(window_length, factor=8)
    k1, k2, k3, k4 = snap_k(k1, k2, k3, k4)
    out1, out2, out3, out4 = snap_channel(out1, out2, out3, out4, factor=8)
    latent_dim    = int(round(latent_dim))
    for SUBJECT_ID in SUBJECT_IDS:
        # ---------- run the committee ----------
        consensus, times_trimmed, events, raw, latent_list, label_streams = committee_for_subject(
            participant_idx = SUBJECT_ID,
            seeds           = SEEDS,
            beta            = float(beta),
            means           = float(means),
            logvar          = float(logvar),
            window_length   = window_length,
            window_buffer   = window_buffer,
            latent_dim      = latent_dim,
            epochs_num      = EPOCHS_NUM,
            verbose         = VERBOSE_MODEL,
            device          = DEVICE,
            k_size          = (k1, k2, k3, k4),
            out_channels    = (out1, out2, out3, out4)
        )
        # ---------- bucket samples  ----------
        onsets = raw.annotations.onset
        descs  = np.asarray(raw.annotations.description)
        event_names = [
            descs[np.where(np.isclose(onsets, ev))[0][0]]
            if np.any(np.isclose(onsets, ev)) else "Unknown"
            for ev in np.sort(events)
        ]

        samples_ctrl, samples_act = [], []
        for (start, end), name in zip(zip(events[:-1], events[1:]),
                                    event_names[:-1]):
            mask = (times_trimmed >= start) & (times_trimmed < end)
            if not mask.any():
                continue
            vals = consensus[mask]
            if "Control" in name:
                samples_ctrl.extend(vals)
            elif "Tapping" in name:
                samples_act.extend(vals)

        if not samples_ctrl or not samples_act:
            # No separation info → poor score
            score += -1
        else:
            score += abs(np.mean(samples_act) - np.mean(samples_ctrl))

        # House-keeping: free GPU
        torch.cuda.empty_cache(); gc.collect()
    return score


# ------------------------------------------------------------------------
# 2.  Search space (edit as you like)                                    |
# ------------------------------------------------------------------------
pbounds = {
    "beta"          : (0.1, 1.0),
    "means"         : (0.5, 3.0),
    "logvar"        : (-3.0, 1.0),   # log(sigma^2)
    "window_length" : (32, 80),      # int, will be rounded
    "window_buffer" : (1.5, 2.5),
    "latent_dim"    : (2, 15),
    "k1"            : (3, 17),
    "k2"            : (3, 17),
    "k3"            : (3, 17),
    "k4"            : (3, 17),
    "out1"          : (16, 128),
    "out2"          : (16, 128),
    "out3"          : (16, 128),
    "out4"          : (16, 128),
}

# ------------------------------------------------------------------------
# 3.  Run Bayesian optimisation                                          |
# ------------------------------------------------------------------------
def main():
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2,
    )

    # Utility function (UCB) for exploration vs. exploitation

    N_INIT = 5    # initial random evals
    N_ITER = 100    # Bayesian-opt steps


    # ------------------------------------------------------------------
    # tqdm progress-bar that advances once per completed optimisation step
    # ------------------------------------------------------------------
    pbar       = tqdm(total=N_INIT + N_ITER, desc="BayesOpt")
    subscriber = TqdmAndLogger(pbar, logfile="bayes_progress.csv")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, subscriber)


    t0 = time()
    optimizer.maximize(
        init_points=N_INIT,
        n_iter=N_ITER,
    )
    t1 = time()

    print("\nBest parameters found:")
    print(optimizer.max)
    print(f"Total optimisation time: {(t1 - t0)/60:.1f} min")

if __name__ == "__main__":
    main()
