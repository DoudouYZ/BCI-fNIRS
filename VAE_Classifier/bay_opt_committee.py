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
# ------------------------------------------------------------------
#  Resume BayesOpt from a CSV log
# ------------------------------------------------------------------
def warm_start_from_csv(optimizer: BayesianOptimization,
                        csv_path: str,
                        verbose: bool = True):
    """
    Load previous evaluations from *csv_path* (same format written by
    TqdmAndLogger) and register them with *optimizer* so that you can
    resume an interrupted Bayesian‐optimisation run.

    Parameters
    ----------
    optimizer : BayesianOptimization
        The fresh optimiser instance you just created.
    csv_path : str
        Path to the CSV file you logged in the previous run.
    verbose : bool
        Print how many points were loaded / skipped.
    """
    if not os.path.isfile(csv_path):
        if verbose:
            print(f"[warm-start] No logfile found at {csv_path}.")
        return

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        loaded = 0
        skipped = 0
        for row in reader:
            target = float(row["target"])
            # strip the 'param_' prefix to get back the true pbounds keys
            params = {k.replace("param_", ""): float(v)
                      for k, v in row.items() if k.startswith("param_")}

            try:
                optimizer.register(params=params, target=target)
                loaded += 1
            except KeyError:
                # the exact point already exists in the internal data structure
                skipped += 1

    if verbose:
        print(f"[warm-start] Registered {loaded} previous points "
              f"({skipped} duplicates skipped).")

def snap_length(L, factor=8, min_len=16):
    L = int(round(L / factor)) * factor
    return max(L, min_len)

def snap_k(k1, k2, k3, k4, k5):
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
    if k4 % 2 == 0:
        k4 += 1
    k5 = int(k5)
    if k5 % 2 == 0:
        k5 += 1
    return (k1, k2, k3, k4, k5)

def snap_channel(o1, o2, o3, o4, o5, factor = 8):
    o1 = int(round(o1 / factor)) * factor
    o2 = int(round(o2 / factor)) * factor
    o3 = int(round(o3 / factor)) * factor
    o4 = int(round(o4 / factor)) * factor
    o5 = int(round(o5 / factor)) * factor
    return (o1, o2, o3, o4, o5)

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
SUBJECT_IDS      = (0,1,2,3,4)
SEEDS           = range(1)      # committee size
EPOCHS_NUM      = 40                   # training epochs per VAE
VERBOSE_MODEL   = False                 # print VAE training logs?
LOAD_SAVED      = False
LOGFILE         = "bayes_wasserstein_all_patients.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=RuntimeWarning)  # NaN warnings


# ------------------------------------------------------------------------
# 1.  Define an objective: maximise cluster *separation*                 |
# ------------------------------------------------------------------------
def objective(means, logvar,
             window_length, window_buffer, latent_dim,
             lam_mmd, mmd_sigma,      # ← NEW if you want to tune them
             k1, k2, k3, k4, k5, out1, out2, out3, out4, out5):

    score = 0.0

    # ---- snap integer-like params --------------------------------
    window_length = snap_length(window_length, factor=8)
    kernel_sizes  = snap_k(k1, k2, k3, k4, k5)
    out_channels  = snap_channel(out1, out2, out3, out4, out5, factor=8)
    latent_dim    = int(round(latent_dim))

    for subj in SUBJECT_IDS:

        results = committee_for_subject(
            participant_idx = subj,
            seeds           = SEEDS,
            means           = float(means),
            logvar          = float(logvar),
            window_length   = window_length,
            window_buffer   = window_buffer,
            latent_dim      = latent_dim,
            epochs_num      = EPOCHS_NUM,
            use_mmd        = True,              # fix or expose as pbounds
            lam_mmd        = lam_mmd,           # tuned
            mmd_sigma      = mmd_sigma,         # tuned
            verbose         = VERBOSE_MODEL,
            device          = DEVICE,
            k_size          = kernel_sizes,
            out_channels    = out_channels
        )

        # -----------------------------------------------------------
        # loop over "right" and "left" sub-dicts
        # -----------------------------------------------------------
        for side in ("right", "left"):

            cons   = results[side]["consensus"]
            times  = results[side]["times"]
            events = np.sort(results[side]["events"])
            raw    = results[side]["raw"]

            onsets = raw.annotations.onset
            descs  = np.asarray(raw.annotations.description)
            names  = [descs[np.where(np.isclose(onsets, ev))[0][0]]
                      if np.any(np.isclose(onsets, ev)) else "Unknown"
                      for ev in events]

            ctrl_vals, task_vals = [], []
            for (s,e), name in zip(zip(events[:-1], events[1:]), names[:-1]):
                mask = (times >= s) & (times < e)
                if not mask.any():
                    continue
                vals = cons[mask]
                if "Control" in name:
                    ctrl_vals.extend(vals)
                elif "Tapping" in name:          # covers Left or Right
                    task_vals.extend(vals)

            if ctrl_vals and task_vals:
                score += np.log(abs(np.mean(task_vals) - np.mean(ctrl_vals))*10 + 0.2)
            else:
                score += -2   # penalise missing separation

        torch.cuda.empty_cache(); gc.collect()

    return score

# ------------------------------------------------------------------------
# 2.  Search space (edit as you like)                                    |
# ------------------------------------------------------------------------
pbounds = {
    "lam_mmd"       : (5, 100),
    "mmd_sigma"     : (1,5),
    "means"         : (0.5, 3.0),
    "logvar"        : (-3.0, 1.0),   # log(sigma^2)
    "window_length" : (32, 160),      # int, will be rounded
    "window_buffer" : (1.5, 2.5),
    "latent_dim"    : (2, 20),
    "k1"            : (3, 17),
    "k2"            : (3, 17),
    "k3"            : (3, 17),
    "k4"            : (3, 17),
    "k5"            : (3, 17),
    "out1"          : (16, 128),
    "out2"          : (16, 128),
    "out3"          : (16, 128),
    "out4"          : (16, 128),
    "out5"          : (16, 128),
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
    if LOAD_SAVED:
        if not os.path.isfile(LOGFILE):
            raise FileNotFoundError(f"LOAD_SAVED=True but '{LOGFILE}' not found.")
        warm_start_from_csv(optimizer, LOGFILE, True)
    # Utility function (UCB) for exploration vs. exploitation

    N_INIT = 20    # initial random evals
    N_ITER = 200    # Bayesian-opt steps


    # ------------------------------------------------------------------
    # tqdm progress-bar that advances once per completed optimisation step
    # ------------------------------------------------------------------
    pbar       = tqdm(total=N_INIT + N_ITER, desc="BayesOpt")
    subscriber = TqdmAndLogger(pbar, logfile=LOGFILE)
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
