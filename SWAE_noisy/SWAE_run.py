"""
Run experiments over multiple subjects, conditions, and 5 seeds per subject.
Conduct experiments for:
  1. Tapping_Left vs Control
  2. Tapping_Right vs Control
  3. Control-only test (split Control into two groups)
Results are saved to results.csv.
"""

import csv
import sys
import torch
import numpy as np
from pathlib import Path
import mne
from torch.utils.data import DataLoader
from scipy.stats import fisher_exact
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from SWAE_preprocessing import preprocess_epochs
from SWAE_model import EpochDataset, Encoder1D, Decoder1D, GMMPrior, sliced_wasserstein

# Ensure root is on sys.path to access Preprocessing module
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data   

# ───────────────────────────────────  HYPER-PARAMS  ───────────────────────────────────
SUBJECTS = [0,1,2,3,4]  
SEEDS = [0,1,2,3,4]  # five seeds per subject

# Conditions to test per subject:
# "left": Tapping_Left vs Control
# "right": Tapping_Right vs Control
# "control_only": Split control epochs into two groups
CONDITIONS = ["control_only", "left", "right"]

N_AUG         = 5     # augmented copies per epoch
NOISE_FACTOR  = 3.0
BATCH_SIZE    = 16
EPOCHS        = 500
Z_DIM         = 8
LR            = 1e-3
LAMBDA_SWD    = 1
WARMUP_EPOCHS = 50
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

results = []

for subj in SUBJECTS:
    for cond in CONDITIONS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"\n=== Running subject {subj}, condition {cond}, seed {seed} ===")
            
            # if cond == "control_only":
            #     # Use full epochs with a defined time window and split control epochs into two groups
            #     full_epochs = get_raw_subject_data(subj, tmin=-30, tmax=30)
            #     control_epochs = full_epochs["Control"]
            #     new_control_epochs = []
            #     # Here we split each 60-sec epoch into two segments (e.g., first 20 sec and last 20 sec)
            #     sfreq = full_epochs["Control"].info['sfreq']
            #     for epoch in control_epochs:
            #         epoch_first = epoch[:, :int(20 * sfreq)]
            #         epoch_last  = epoch[:, -int(20 * sfreq):]
            #         new_control_epochs.extend([epoch_first, epoch_last])
            #     idx = np.random.permutation(len(new_control_epochs))
            #     split = len(idx)//2
            #     control_A_list = [new_control_epochs[i] for i in idx[:split]]
            #     control_B_list = [new_control_epochs[i] for i in idx[split:]]
            #     # Build mne.EpochsArray (using Control info) for each split
            #     info = full_epochs["Control"].info.copy()
            #     control_A = mne.EpochsArray(np.stack(control_A_list), info, tmin=-30)
            #     control_B = mne.EpochsArray(np.stack(control_B_list), info, tmin=-30)
            #     epochs_dict = {"Control_A": control_A, "Control_B": control_B}
            #     labels = list(epochs_dict)
            # else:
            #     # For left/right hands, use tapping epochs vs Control
            #     full = get_raw_subject_data(subj)
            #     tap_label = "Tapping_Left" if cond == "left" else "Tapping_Right"
            #     epochs_dict = {tap_label: full[tap_label], "Control": full["Control"]}
            #     labels = list(epochs_dict)
            if cond == "control_only":
                full = get_raw_subject_data(subj)
                ctl = full["Control"]
                idx = np.random.permutation(len(ctl))
                A,B = np.array_split(idx, 2)
                epochs_dict = {"Control_A": ctl[A], "Control_B": ctl[B]}
                labels = list(epochs_dict)
            else:
                # tap = "Tapping_Left" if USE_LEFT_HAND else "Tapping_Right"
                # epochs_dict = {tap: full[tap], "Control": full["Control"]}
                full = get_raw_subject_data(subj)
                tap_label = "Tapping_Left" if cond == "left" else "Tapping_Right"
                epochs_dict = {tap_label: full[tap_label], "Control": full["Control"]}
                labels = list(epochs_dict)
            
            print(f"Epoch counts: { {l: len(epochs_dict[l]) for l in labels} }")
            arrays = preprocess_epochs(epochs_dict, n_aug=N_AUG, rng_seed=seed)
            for k, v in arrays.items():
                print(f"{k}: {len(v)} samples")
            
            all_samples = arrays[labels[0]] + arrays[labels[1]]
            ds = EpochDataset(all_samples)
            dl = DataLoader(ds, BATCH_SIZE, shuffle=True, drop_last=True)
            print(f"Input shape  C={ds.C}, T={ds.T}, total {len(ds)} samples")
            
            # ───────────────────────────────────  MODEL  ───────────────────────────────────
            enc = Encoder1D(ds.C, Z_DIM).to(DEVICE)
            dec = Decoder1D(ds.C, ds.T, Z_DIM).to(DEVICE)
            pri = GMMPrior(Z_DIM).to(DEVICE)
            
            opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(pri.parameters()), lr=LR)
            mse = torch.nn.MSELoss()
            
            # ───────────────────────────────────  TRAINING  ───────────────────────────────────
            enc.train()
            dec.train()
            pri.train()
            for ep in range(1, EPOCHS+1):
                if ep < 10:
                    current_lambda = 0
                elif ep < WARMUP_EPOCHS:
                    current_lambda = LAMBDA_SWD * ((ep-10) / (WARMUP_EPOCHS-10))
                else:
                    current_lambda = LAMBDA_SWD
                rec_tot, swd_tot = 0., 0.
                for xb in dl:
                    xb = xb.to(DEVICE)
                    mu, logv = enc(xb)
                    z = mu + NOISE_FACTOR * torch.exp(0.5 * logv) * torch.randn_like(mu)
                    xh = dec(z)
                    loss_rec = mse(xh, xb)
                    loss_swd = sliced_wasserstein(z, pri.sample(len(z)), n_proj=400)
                    loss = loss_rec + current_lambda * loss_swd
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    rec_tot += loss_rec.item()
                    swd_tot += loss_swd.item()
                if ep % 50 == 0 or ep == 1:
                    print(f"Subj {subj} Cond {cond} Seed {seed} Ep {ep:3d}: recon {rec_tot/len(dl):.4e}  swd {swd_tot/len(dl):.4e}")
            
            # ───────────────────────────────────  CLUSTER TEST  ───────────────────────────────────
            def embed(arrs):
                enc.eval()
                out = []
                with torch.no_grad():
                    for a in DataLoader(EpochDataset(arrs), batch_size=256):
                        mu, _ = enc(a.to(DEVICE))
                        out.append(mu.cpu())
                return torch.cat(out)
            
            lat_A = embed(arrays[labels[0]])
            lat_B = embed(arrays[labels[1]])
            
            def assign(z):
                device = z.device
                mu  = pri.mu.detach().to(device).squeeze()
                var = torch.exp(pri.logvar.detach().to(device).squeeze())
                d0 = ((z - mu[0])**2 / var[0]).sum(1)
                d1 = ((z - mu[1])**2 / var[1]).sum(1)
                return (d1 < d0).long()
            
            cont = np.zeros((2,2), int)
            cont[0] += np.bincount(assign(lat_A), minlength=2)
            cont[1] += np.bincount(assign(lat_B), minlength=2)
            p_val = fisher_exact(cont)[1]
            print(f"Subj {subj} Cond {cond} Seed {seed} Fisher p = {p_val:.4e}")
            
            results.append({
                "subject": subj,
                "condition": cond,
                "seed": seed,
                "fisher_p": p_val,
                "contingency": cont.tolist()
            })

# Save results to CSV
csv_file = Path("results.csv")
with csv_file.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["subject", "condition", "seed", "fisher_p", "contingency"])
    writer.writeheader()
    for res in results:
        writer.writerow(res)
print(f"\nResults saved to {csv_file.resolve()}")