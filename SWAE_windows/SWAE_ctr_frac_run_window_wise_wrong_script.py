import os
import sys
import json
import numpy as np
import pandas as pd
import mne
import torch
import matplotlib.pyplot as plt
from scipy.stats import binomtest, f
from numpy.random import default_rng
from scipy.stats import fisher_exact

# Insert parent folder for custom modules (Preprocessing and PA classification modules)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preprocessing import get_raw_subject_data
from PA_classifier.PA_tests.pa_classification_run_no_overlap import replace_fraction_with_control_no_overlap_both

# Import SWAE functions and models
from SWAE_preprocessing import preprocess_for_wae_windows
from SWAE_model import WindowDataset, build_dataloader, Encoder2D, Decoder2D, GMMPrior, sliced_wasserstein

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Hyper-parameters ─────────────────────────────────────────────
SUBJECTS         = [0, 1, 2, 3, 4]
FRACTIONS        = np.arange(0.00, 0.45 + 0.05, 0.05)
INNER_SEEDS      = range(20)    # inner seeds per combined test (SWAE run)

WINDOW_SIZE      = 100
WINDOW_STEP      = 20
BATCH_SIZE       = 64
EPOCHS           = 140
Z_DIM            = 11
LR               = 5e-4
LAMBDA_SWD       = 0.7
WARMUP_EPOCHS    = 50
FIRST_EPOCHS     = 50
N_PROJ           = 465
PRIOR_MEAN       = 1.0
PRIOR_LOGVAR     = 0.0
LEARNABLE_PRIOR  = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Helper functions ─────────────────────────────────────────────

def centroid_mahalanobis(X, y):
    X1, X2 = X[y == 0], X[y == 1]
    S = np.cov(np.vstack([X1, X2]).T)
    diff = X1.mean(0) - X2.mean(0)
    D2 = diff @ np.linalg.inv(S) @ diff
    return np.sqrt(D2)

def hotellings_t2(X, y):
    X1, X2 = X[y == 0], X[y == 1]
    n1, p = X1.shape
    n2 = X2.shape[0]
    mean1 = X1.mean(0)
    mean2 = X2.mean(0)
    S1 = np.cov(X1, rowvar=False)
    S2 = np.cov(X2, rowvar=False)
    Sp = ((n1 - 1)*S1 + (n2 - 1)*S2) / (n1 + n2 - 2)
    diff = mean1 - mean2
    T2 = (n1 * n2) / (n1 + n2) * diff @ np.linalg.inv(Sp) @ diff
    F_stat = (n1 + n2 - p - 1) * T2 / ((n1 + n2 - 2) * p)
    df1, df2 = p, (n1 + n2 - p - 1)
    p_val = 1 - f.cdf(F_stat, df1, df2)
    return T2, F_stat, p_val

def assign(z, pri):
    # assign based on closest prior mode using Mahalanobis distance
    diff = z.unsqueeze(1) - pri.mu
    d2 = (diff**2 / torch.exp(pri.logvar).unsqueeze(0)).sum(dim=2)
    return torch.argmin(d2, dim=1).cpu().numpy()

def run_swae_for_side(tap_epochs, control_epochs, inner_seed, side_label):
    """
    Runs one SWAE simulation for one tapping side vs control.
    Returns True if the run passes the thresholds, False otherwise.
    """
    # Set seeds for reproducibility per simulation
    torch.manual_seed(inner_seed)
    np.random.seed(inner_seed)

    # Define labels and create epoch dictionary.
    # side_label should be "Tapping_Left" or "Tapping_Right"
    epochs_dict = {side_label: tap_epochs, "Control": control_epochs}
    labels = [side_label, "Control"]

    # Preprocess: extract sliding windows from epochs.
    mats = preprocess_for_wae_windows(epochs_dict, class_labels=labels, 
                                       window_size=WINDOW_SIZE, window_step=WINDOW_STEP)
    all_mats = mats[labels[0]] + mats[labels[1]]
    # Adjust batch size if needed.
    bsize = BATCH_SIZE
    if bsize > len(mats[labels[0]]):
        bsize = len(mats[labels[0]])

    ds = WindowDataset(all_mats)
    C, T = ds.C, ds.T

    # Build model, optimizer and loss
    enc = Encoder2D(C, Z_DIM).to(DEVICE)
    dec = Decoder2D(C, T, Z_DIM).to(DEVICE)
    pri = GMMPrior(Z_DIM, mu=PRIOR_MEAN, logvar=PRIOR_LOGVAR, learnable=LEARNABLE_PRIOR).to(DEVICE)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(pri.parameters()), lr=LR)
    mse = torch.nn.MSELoss()

    dl = build_dataloader(all_mats, bsize, shuffle=True, drop_last=True)

    # Training loop
    for ep in range(1, EPOCHS+1):
        if ep < FIRST_EPOCHS:
            current_lambda = 0.05
        elif ep < WARMUP_EPOCHS:
            current_lambda = LAMBDA_SWD * ((ep - FIRST_EPOCHS) / (WARMUP_EPOCHS - FIRST_EPOCHS))
        else:
            current_lambda = LAMBDA_SWD

        for xb in dl:
            xb = xb.to(DEVICE)
            mu_enc, logv_enc = enc(xb)
            z = mu_enc + torch.exp(0.5*logv_enc) * torch.randn_like(mu_enc)
            xb_hat = dec(z)
            l_rec = mse(xb_hat, xb)
            l_swd = sliced_wasserstein(z, pri.sample(xb.size(0)), n_proj=N_PROJ)
            loss = l_rec + current_lambda * l_swd

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Clamp the learnable prior parameters
            pri.clamp_parameters(mu_min=-2.0, mu_max=2.0, logvar_min=-5.0, logvar_max=5.0)

    # Redefine assign to use the trained prior from SWAE training
    def assign(z):
        diff = z.unsqueeze(1) - pri.mu
        d2 = (diff**2 / torch.exp(pri.logvar).unsqueeze(0)).sum(dim=2)
        return torch.argmin(d2, dim=1).cpu().numpy()

    table = np.zeros((2,2), int)
    enc.eval()
    for r,lbl in enumerate(labels):
        dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
        with torch.no_grad():
            for xb in dl_lbl:
                cl = assign(enc(xb.to(DEVICE))[0])
                table[r] += np.bincount(cl, minlength=2)

    p_fish = fisher_exact(table)[1]

    # Evaluation on window-level predictions
    all_lat = []
    all_labels = []
    for lbl in labels:
        dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
        with torch.no_grad():
            for xb in dl_lbl:
                out = enc(xb.to(DEVICE))[0]
                all_lat.append(out.cpu())
                all_labels.extend([lbl]*xb.size(0))
    lat = torch.cat(all_lat).numpy()
    label_int = np.array([0 if l == side_label else 1 for l in all_labels])
    
    # Compute centroid Mahalanobis distance
    D = centroid_mahalanobis(lat, label_int)
    # Silhouette score
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(lat, label_int)
    rng = np.random.default_rng(inner_seed)
    sil_null = []
    for _ in range(500):
        perm_labels = rng.permutation(label_int)
        sil_null.append(silhouette_score(lat, perm_labels))
    sil_thresh = np.percentile(sil_null, 95)
    
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.utils import shuffle as sk_shuffle

    clf = SVC(kernel="linear")
    acc = cross_val_score(clf, lat, label_int, cv=5).mean()
    perm_acc = np.array([
        cross_val_score(clf, lat, sk_shuffle(label_int), cv=5).mean()
        for _ in range(1000)
    ])
    p_val_perm = (perm_acc >= acc).mean()
    
    # Check criteria – using either of the threshold criteria
    passed = (p_val_perm < 0.05) or (p_fish < 0.05)
    return passed

# ─── MAIN LOOP ───────────────────────────────────────────────────
results_all = []  # To store individual inner seed booleans per subject and fraction

for subj in SUBJECTS:
    for frac in FRACTIONS:
        print(f"Running subject {subj}, fraction {frac:.2f}...")
        # Load original epochs for subject
        epochs = get_raw_subject_data(subject=subj, tmin=-5, tmax=15)
        # Use a fixed seed for control replacement since we removed outer seeds
        tap_left_mixed, tap_right_mixed, new_control = replace_fraction_with_control_no_overlap_both(
            epochs["Tapping_Left"], epochs["Tapping_Right"], epochs["Control"], frac, seed=0)
        
        # Run SWAE simulation for each inner seed and for each tapping side.
        left_results = []
        for inner_seed in INNER_SEEDS:
            flag = run_swae_for_side(tap_left_mixed, new_control, inner_seed, side_label="Tapping_Left")
            print(f"Subject {subj}, frac {frac:.2f}, Tapping_Left, inner seed {inner_seed}: {flag}")
            left_results.append(flag)
            
        right_results = []
        for inner_seed in INNER_SEEDS:
            flag = run_swae_for_side(tap_right_mixed, new_control, inner_seed, side_label="Tapping_Right")
            print(f"Subject {subj}, frac {frac:.2f}, Tapping_Right, inner seed {inner_seed}: {flag}")
            right_results.append(flag)
            
        results_all.append({
            "subject": subj,
            "CONTROL_REPLACEMENT_FRAC": frac,
            "Tapping_Left": left_results,
            "Tapping_Right": right_results
        })

# Save overall results to JSON file
with open("swae_inner_seed_results.json", "w") as f:
    json.dump(results_all, f, indent=4)

print("Overall inner seed results saved to swae_inner_seed_results.json")