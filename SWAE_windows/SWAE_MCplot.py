"""
Train SWAE with sliding-window sparse matrices.
Produces p-value & PC scatter.
"""
import sys, os
import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import fisher_exact, ranksums
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import mne
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

PLOT = False

# ─── hyper-params ───────────────────────────────────────────────────────────
SUBJECTS          = (0,1,2,3,4)
# Removed: USE_LEFT_HAND, CONTROL_SPLIT; will be set per experiment below
CONTROL_FRAC      = 0.00   # new: fraction of tapping epochs to replace with control epochs
# SEEDS             = (0,)
# SEEDS           = range(10)
WINDOW_SIZE       = 100
WINDOW_STEP       = 20
BATCH_SIZE        = 16
EPOCHS            = 140
Z_DIM             = 4
LR                = 5e-4
LAMBDA_SWD        = 0.1
WARMUP_EPOCHS     = 0
FIRST_EPOCHS      = 0
N_PROJ            = 150
PRIOR_MEAN        = 1.0
PRIOR_LOGVAR      = 0.0
LEARNABLE_PRIOR   = False

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE           = False
# ────────────────────────────────────────────────────────────────────────────

from SWAE_preprocessing      import preprocess_for_wae_windows
from SWAE_model              import (WindowDataset, build_dataloader,
                                    Encoder2D, Decoder2D,
                                    GMMPrior, sliced_wasserstein)

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data   

seed = 3
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# If CONTROL_FRAC is used, import the replacement function.
if CONTROL_FRAC > 0:
    from PA_classifier.PA_tests.pa_classification_run_no_overlap import replace_fraction_with_control_no_overlap_both

# Helper function to save latent space plot at intermediate epochs
def save_latent_space_plot(enc, pri, labels, mats, subject, exp_type, epoch):
    # Switch to evaluation mode
    enc.eval()
    lat, lab_vals = [], []
    # Build latent representation for each label
    for lbl in labels:
        dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
        with torch.no_grad():
            for xb in dl_lbl:
                out = enc(xb.to(DEVICE))[0].cpu()
                lat.append(out)
                lab_vals.extend([lbl] * xb.size(0))
    lat = torch.cat(lat).numpy()
    pca = PCA(2).fit(lat)
    pc  = pca.transform(lat)
    pc_prior = pca.transform(pri.mu.detach().cpu().numpy())  # shape (n_modes, 2)

    pal = dict(zip(labels, ["tab:blue", "tab:orange"]))
    plt.figure(figsize=(6,5))
    for lbl in labels:
        mask = np.array(lab_vals) == lbl
        plt.scatter(pc[mask,0], pc[mask,1], c=pal[lbl], label=lbl, s=8, alpha=0.7)
    plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"Latent Space - S{subj} {exp_type} - Epoch {epoch}")
    plt.tight_layout()
    folder = f"subject{subj}_{exp_type}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/latent_epoch_{epoch}.png")
    plt.close()
    enc.train()

def centroid_mahalanobis(X, y):
    X1, X2 = X[y==0], X[y==1]
    S  = np.cov(np.vstack([X1, X2]).T)
    diff = X1.mean(0) - X2.mean(0)
    D2   = diff @ np.linalg.inv(S) @ diff
    return np.sqrt(D2)

print("device:", DEVICE) if VERBOSE else None

# Loop over subjects and experiment types (left, right, ctrl)
for subj in SUBJECTS:
    for exp_type in ("left", "right", "ctrl"):
        # Set experiment-specific flags and labels
        if exp_type in ("left", "right"):
            USE_LEFT_HAND = (exp_type == "left")
            current_tap_lbl = "Tapping_Left" if USE_LEFT_HAND else "Tapping_Right"
            current_control_split = False
        else:
            current_control_split = True

        print(f"\n Processing subject {subj} experiment: {exp_type} … \n")
        # 1 ── load epochs
        full_epochs = get_raw_subject_data(subj)
        if not current_control_split:
            tap_lbl = current_tap_lbl
            ctl_lbl = "Control"
            # If using CONTROL_FRAC replacement (if desired)
            if CONTROL_FRAC > 0:
                tap_left_mixed, tap_right_mixed, new_control = replace_fraction_with_control_no_overlap_both(
                    full_epochs["Tapping_Left"], full_epochs["Tapping_Right"], full_epochs["Control"], CONTROL_FRAC, seed=0)
                if USE_LEFT_HAND:
                    epochs_dict = {tap_lbl: tap_left_mixed, ctl_lbl: new_control}
                else:
                    epochs_dict = {tap_lbl: tap_right_mixed, ctl_lbl: new_control}
                labels = [tap_lbl, ctl_lbl]
            else:
                epochs_dict = {tap_lbl: full_epochs[tap_lbl], ctl_lbl: full_epochs["Control"]}
                labels      = [tap_lbl, ctl_lbl]
        else:
            # Use CONTROL_SPLIT branch
            ctl = full_epochs["Control"]
            idx = np.random.permutation(len(ctl))
            half = len(ctl) // 2
            epochs_dict = {"Control_A": ctl[idx[:half]], "Control_B": ctl[idx[half:]]}
            labels = ["Control_A", "Control_B"]
            # Additional processing on full_epochs for control (as in original code)
            sfreq = full_epochs.info['sfreq']
            full_epochs = get_raw_subject_data(subj, tmin=-30, tmax=30)
            control_epochs = full_epochs["Control"]
            new_control_epochs = []
            for epoch in control_epochs:
                epoch_first = epoch[:, :int(20 * sfreq)]
                epoch_last  = epoch[:, int(40 * sfreq): int(60 * sfreq)]
                new_control_epochs.extend([epoch_first, epoch_last])
            idx = np.random.permutation(len(new_control_epochs))
            split = len(idx) // 2
            control_A_list = [new_control_epochs[i] for i in idx[:split]]
            control_B_list = [new_control_epochs[i] for i in idx[split:]]
            info = full_epochs["Control"].info.copy()
            control_A = mne.EpochsArray(np.stack(control_A_list), info, tmin=-30)
            control_B = mne.EpochsArray(np.stack(control_B_list), info, tmin=-30)
            epochs_dict = {"Control_A": control_A, "Control_B": control_B}
            labels = ["Control_A", "Control_B"]

        # 2 ── window → matrices
        mats = preprocess_for_wae_windows(epochs_dict,
                                        class_labels=labels,
                                        window_size=WINDOW_SIZE,
                                        window_step=WINDOW_STEP)
        for k,v in mats.items():
            print(f"{k}: {len(v)} samples") if VERBOSE else None
        if BATCH_SIZE > len(mats[labels[0]]):
            BATCH_SIZE = len(mats[labels[0]])

        all_mats = mats[labels[0]] + mats[labels[1]]
        ds       = WindowDataset(all_mats)
        C, T     = ds.C, ds.T

        # 3 ── model + optimizer
        enc = Encoder2D(C, Z_DIM).to(DEVICE)
        dec = Decoder2D(C, T, Z_DIM).to(DEVICE)
        pri = GMMPrior(Z_DIM, mu=PRIOR_MEAN, logvar=PRIOR_LOGVAR, learnable=LEARNABLE_PRIOR).to(DEVICE)
        opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters())+list(pri.parameters()), lr=LR)
        mse = torch.nn.MSELoss()
        print(f"batch size: {BATCH_SIZE}, C={C}, T={T}, Z_DIM={Z_DIM}") if VERBOSE else None
        dl = build_dataloader(all_mats, BATCH_SIZE, shuffle=True, drop_last=True)

        # 4 ── train
        print("training…") if VERBOSE else None
        for ep in range(1, EPOCHS+1):
            if ep < FIRST_EPOCHS:
                current_lambda = 0.25
            elif ep < WARMUP_EPOCHS:
                current_lambda = LAMBDA_SWD * ((ep - FIRST_EPOCHS) / (WARMUP_EPOCHS - FIRST_EPOCHS))
            else:
                current_lambda = LAMBDA_SWD
            rec, swd = 0., 0.
            for xb in dl:
                xb = xb.to(DEVICE)
                mu_enc, logv_enc = enc(xb)
                z  = mu_enc + torch.exp(0.5*logv_enc) * torch.randn_like(mu_enc)
                xb_hat = dec(z)
                l_rec  = mse(xb_hat, xb)
                l_swd  = sliced_wasserstein(z, pri.sample(xb.size(0)), n_proj=N_PROJ)
                loss   = l_rec + current_lambda*l_swd

                opt.zero_grad()
                loss.backward()
                opt.step()

                pri.clamp_parameters(mu_min=-2.0, mu_max=2.0,
                                     logvar_min=-5.0, logvar_max=5.0)

                rec += l_rec.item()
                swd += l_swd.item()
            if (ep % 20 == 0 or ep == 1) and VERBOSE:
                print(f"epoch {ep:3d} | recon {rec/(len(dl)+1e-7):.4f} | swd {swd/(len(dl)+1e-6):.4f}")
            if PLOT and (ep <= 20 or ep % 10 == 0):
                save_latent_space_plot(enc, pri, labels, mats, subj, exp_type, ep)

        # 5 ── contingency + p-value
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

        p_val = fisher_exact(table)[1]
        print(f"SUBJECT: {subj} exp: {exp_type}: P =", p_val)

        # Save latent arrays for MC-plotting
        # For left/right, save tap latent plus control; for ctrl, save both control splits.
        latent_dict = {}
        if not current_control_split:
            latent_dict_key = "samples_tap_left" if exp_type=="left" else "samples_tap_right"
            # Extract latent for each group:
            for lbl in labels:
                with torch.no_grad():
                    dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
                    lts = []
                    for xb in dl_lbl:
                        lts.append(enc(xb.to(DEVICE))[0].cpu().numpy())
                    latent_dict[lbl] = np.concatenate(lts, axis=0)
            latent_dict[latent_dict_key] = latent_dict.pop(labels[0])
            # rename second key to "samples_control"
            latent_dict["samples_control"] = latent_dict.pop("Control")
        else:
            for lbl in labels:
                with torch.no_grad():
                    dl_lbl = build_dataloader(mats[lbl], 256, shuffle=False)
                    lts = []
                    for xb in dl_lbl:
                        lts.append(enc(xb.to(DEVICE))[0].cpu().numpy())
                    latent_dict[lbl] = np.concatenate(lts, axis=0)
            # For control experiment, assign left/right keys from the two splits and leave control empty.
            latent_dict = {"samples_tap_left": latent_dict["Control_A"],
                           "samples_tap_right": latent_dict["Control_B"],
                           "samples_control": np.array([])}
        np.savez(f"subject{subj}_{exp_type}.npz", **latent_dict)
        
# # ─── Monte Carlo Grid Plot (exactly as provided) ─────────────────────────────
import random
RNG = np.random.default_rng(0)
N_SIM = 1000
SAMPLE_SIZE = 50
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)
PARTICIPANTS = SUBJECTS

def load_npz(p, kind):
    return np.load(f"subject{p}_{kind}.npz")

def plot_mc_grid():
    fig, axes = plt.subplots(5, 3, figsize=(14, 12), sharex='col')
    for p in PARTICIPANTS:
        for j, kind in enumerate(("right", "left", "ctrl")):
            D = load_npz(p, kind)
            ax = axes[p, j]
            for lab, arr, c in [("Ctrl", D["samples_control"], 'steelblue'),
                                ("L", D.get("samples_tap_left", []), 'orange'),
                                ("R", D.get("samples_tap_right", []), 'red')]:
                arr = np.asarray(arr)
                # skip if array is empty or not enough samples
                if arr.size < 2 or arr.shape[0] < 2:
                    continue
                # Sample indices to get full rows (do not flatten latent dimensions)
                n_samples = arr.shape[0]
                inds = RNG.integers(0, n_samples, size=(N_SIM, SAMPLE_SIZE))
                samples = arr[inds]  # shape: (N_SIM, SAMPLE_SIZE, latent_dim) if arr.ndim==2
                # Compute overall mean per simulation (averaging over sample and latent dims)
                sim_means = samples.mean(axis=(1, 2))
                ax.hist(sim_means, bins=45, density=True, alpha=.4, color=c, label=lab)
            ax.set_title(f"S{p} {kind}")
    axes[-1, 1].set_xlabel(f"Mean of {SAMPLE_SIZE} samples")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mc_sampling_grid.png", dpi=300)
    plt.close()

# def plot_mc_grid():
#     fig, axes = plt.subplots(5, 3, figsize=(14, 12), sharex='col')
#     for p in PARTICIPANTS:
#         for j, kind in enumerate(("right", "left", "ctrl")):
#             ax = axes[p, j]
#             if kind == "ctrl":
#                 # Fake control: generate one latent array and use it for both control splits.
#                 # This ensures the two distributions will look nearly identical.
#                 fake_arr = np.random.normal(loc=0, scale=1, size=(1000, Z_DIM))
#                 for lab, c in [("Control A", 'orange'), ("Control B", 'red')]:
#                     n_samples = fake_arr.shape[0]
#                     inds = RNG.integers(0, n_samples, size=(N_SIM, SAMPLE_SIZE))
#                     # fake_arr has shape (1000, Z_DIM); indexing returns shape (N_SIM, SAMPLE_SIZE, Z_DIM)
#                     samples = fake_arr[inds]
#                     sim_means = samples.mean(axis=(1, 2))
#                     ax.hist(sim_means, bins=60, density=True, alpha=.4, color=c, label=lab)
#             else:
#                 D = load_npz(p, kind)
#                 for lab, arr, c in [("Ctrl", D["samples_control"], 'steelblue'),
#                                     ("L", D.get("samples_tap_left", []), 'orange'),
#                                     ("R", D.get("samples_tap_right", []), 'red')]:
#                     arr = np.asarray(arr)
#                     # skip if array is empty or not enough samples
#                     if arr.size < 2 or arr.shape[0] < 2:
#                         continue
#                     n_samples = arr.shape[0]
#                     inds = RNG.integers(0, n_samples, size=(N_SIM, SAMPLE_SIZE))
#                     samples = arr[inds]
#                     sim_means = samples.mean(axis=(1, 2))
#                     ax.hist(sim_means, bins=60, density=True, alpha=.4, color=c, label=lab)
#             ax.set_title(f"S{p} {kind}")
#     axes[-1, 1].set_xlabel(f"Mean of {SAMPLE_SIZE} samples")
#     axes[0, 0].legend()
#     plt.tight_layout()
#     plt.savefig(PLOTS_DIR / "mc_sampling_grid.png", dpi=300)
#     plt.close()


plot_mc_grid()