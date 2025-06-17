
import sys
import torch, numpy as np
from pathlib import Path
from scipy.stats import fisher_exact
from torch.utils.data import DataLoader
import mne
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ────────────────────────────────────────────────────────────────────────────
#  HYPER-PARAMETERS
# ────────────────────────────────────────────────────────────────────────────
SUBJECT_INDEX        = 0
USE_LEFT_HAND        = False        # ignored if CONTROL_SANITY_CHECK = True
CONTROL_SANITY_CHECK = False       # ← toggle this for the “two-control” run
N_PERM               = 1
MAX_PERM             = 5_000
BATCH_SIZE           = 32
EPOCHS               = 200
Z_DIM                = 2
LEARNING_RATE        = 1e-3
LAMBDA_SWD           = 25
WARMUP_EPOCHS        = 20  # number of epochs over which to ramp up lambda
N_PROJ               = 350
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", DEVICE)

# ────────────────────────────────────────────────────────────────────────────
#  IMPORT UTILITIES (assumed to be in PYTHONPATH)
# ────────────────────────────────────────────────────────────────────────────
from WAE_preprocessing  import preprocess_for_wae_cube
from WAE_model          import CubeEncoder, CubeDecoder, GMMPrior, sliced_wasserstein, CubeDataset

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data

torch.manual_seed(0)
np.random.seed(0)

def cluster_assignment(z, prior):
    """
    z      : (B, z_dim) tensor
    prior  : GMMPrior instance (with .mu, .logvar)
    Returns: (B,) int tensor of cluster indices (0 or 1)
    """
    mu  = prior.mu.detach()             # (2, z_dim)
    var = torch.exp(prior.logvar.detach())
    diff = z.unsqueeze(1) - mu          # (B, 2, z_dim)
    d2   = (diff**2 / var.unsqueeze(0)).sum(dim=2)   # Mahalanobis
    return torch.argmin(d2, dim=1)      # (B,)


# ────────────────────────────────────────────────────────────────────────────
#  1. Load & slice epochs
# ────────────────────────────────────────────────────────────────────────────
full_epochs = get_raw_subject_data(SUBJECT_INDEX)

sfreq = full_epochs.info['sfreq']
if CONTROL_SANITY_CHECK:
    # Load 60-second epochs (as mne.Epochs or numpy arrays)
    full_epochs = get_raw_subject_data(SUBJECT_INDEX, tmin=-30, tmax=30)
    control_epochs = full_epochs["Control"]
    new_control_epochs = []  # will contain two epochs per original epoch
    for epoch in control_epochs:
        # If epoch is a NumPy array of shape (n_channels, n_times) then slice manually:
        # Calculate sample indices (assuming time axis corresponds to the second dimension)
        # For first 20 seconds: from -30 to -10 seconds
        epoch_first = epoch[:, :int(20 * sfreq)]
        # For last 20 seconds: from 10 to 30 seconds (i.e. last 20 seconds)
        epoch_last  = epoch[:, int(40 * sfreq): int(60 * sfreq)]
        new_control_epochs.extend([epoch_first, epoch_last])
    
    # Randomly split the doubled epochs into two groups
    idx = np.random.permutation(len(new_control_epochs))
    split = len(idx) // 2
    control_A_list = [new_control_epochs[i] for i in idx[:split]]
    control_B_list = [new_control_epochs[i] for i in idx[split:]]
    # If needed, convert lists of arrays back to mne.Epochs using mne.EpochsArray
    # Here we assume the info is available from full_epochs["Control"]
    info = full_epochs["Control"].info.copy()
    control_A = mne.EpochsArray(np.stack(control_A_list), info, tmin=-30)
    control_B = mne.EpochsArray(np.stack(control_B_list), info, tmin=-30)
    epochs_dict = {"Control_A": control_A, "Control_B": control_B}
    class_labels = ["Control_A", "Control_B"]
else:
    tap_lbl = "Tapping_Left" if USE_LEFT_HAND else "Tapping_Right"
    ctl_lbl = "Control"
    epochs_dict  = {tap_lbl: full_epochs[tap_lbl],
                    ctl_lbl: full_epochs[ctl_lbl]}
    class_labels = [tap_lbl, ctl_lbl]

print("Epoch counts:",
      {lbl: len(ep) for lbl, ep in epochs_dict.items()})

# ────────────────────────────────────────────────────────────────────────────
#  2. Build permutation cubes
# ────────────────────────────────────────────────────────────────────────────
cubes_per_label = preprocess_for_wae_cube(
    epochs_dict,
    class_labels=class_labels,
    n_perm=N_PERM,
    max_perm=MAX_PERM,
    rng=0,
)

for k, v in cubes_per_label.items():
    print(f"   {len(v):5d} cubes from {k}")

# merge the two labels for unsupervised training
cubes_all = cubes_per_label[class_labels[0]] + cubes_per_label[class_labels[1]]

# ────────────────────────────────────────────────────────────────────────────
#  3. Dataset / DataLoader
# ────────────────────────────────────────────────────────────────────────────
ds = CubeDataset(cubes_all, zscore=True)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                 shuffle=True, drop_last=True)
print(f"Cube shape: (n_perm={ds.D}, C={ds.C}, T={ds.T})")

# ────────────────────────────────────────────────────────────────────────────
#  4. Build model
# ────────────────────────────────────────────────────────────────────────────
enc = CubeEncoder(ds.D, ds.C, z_dim=Z_DIM).to(DEVICE)
dec = CubeDecoder(ds.D, ds.C, ds.T, z_dim=Z_DIM).to(DEVICE)
pri = GMMPrior(Z_DIM).to(DEVICE)

opt = torch.optim.Adam(
    list(enc.parameters()) + list(dec.parameters()) + list(pri.parameters()),
    lr=LEARNING_RATE
)
mse = torch.nn.MSELoss()

# ────────────────────────────────────────────────────────────────────────────
#  5. Train
# ────────────────────────────────────────────────────────────────────────────
print("\n===== Training SWAE =====")
hist = {"recon": [], "swd": []}

for ep in range(1, EPOCHS + 1):
    # Linearly increase lambda during the warm-up period
    if ep < 10:
        current_lambda = 0
    elif ep < WARMUP_EPOCHS:
        current_lambda = LAMBDA_SWD * ((ep-10) / (WARMUP_EPOCHS-10))
    else:
        current_lambda = LAMBDA_SWD

    rec_sum, swd_sum = 0.0, 0.0
    for x in dl:
        x = x.to(DEVICE)
        mean, log_sigma_2 = enc(x)
        clamped_log_sigma2 = torch.clamp(log_sigma_2, min=-12, max=12)
        z = mean + torch.exp(0.5 * clamped_log_sigma2) * torch.randn_like(mean)
        x_hat = dec(z)

        l_rec = mse(x_hat, x)
        l_swd = sliced_wasserstein(z, pri.sample(x.size(0)), n_proj=N_PROJ)
        loss  = l_rec + current_lambda * l_swd

        opt.zero_grad()
        loss.backward()
        opt.step()

        rec_sum += l_rec.item()
        swd_sum += l_swd.item()

    hist["recon"].append(rec_sum / len(dl))
    hist["swd"].append(swd_sum / len(dl))
    if ep % 10 == 0 or ep == EPOCHS:
        print(f"Epoch {ep:3d} | lambda {current_lambda:.4f} | recon {hist['recon'][-1]:.5f} | swd {hist['swd'][-1]:.5f}")
    
# ──────────────────────────────────────────────────────────────────────────
#  5b.  Latent-space separation test  (Control vs Tapping)
# ──────────────────────────────────────────────────────────────────────────
encoder = enc.eval()                    # switch to eval mode
contingency = np.zeros((2, 2), dtype=int)   # rows=labels, cols=clusters

label_names = class_labels              # same order as earlier loop

for row, lbl in enumerate(label_names):
    ds_lbl = CubeDataset(cubes_per_label[lbl], zscore=True)
    dl_lbl = DataLoader(ds_lbl, batch_size=256, shuffle=False)
    with torch.no_grad():
        for xb in dl_lbl:
            xb = xb.to(DEVICE)
            mean, _ = encoder(xb)          # (B, z_dim)
            cl  = cluster_assignment(mean, pri).cpu().numpy()
            for c in cl:
                contingency[row, c] += 1

print("\nContingency table (rows: labels, cols: clusters):")
print(contingency)

odds, p_value = fisher_exact(contingency)
print(f"P-value (Fisher’s exact) = {p_value:.4e}")


# ────────────────────────────────────────────────────────────────────────────
#  6. Save checkpoint
# ────────────────────────────────────────────────────────────────────────────
ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
tag = ("ctlsplit" if CONTROL_SANITY_CHECK else
       ("left" if USE_LEFT_HAND else "right"))
ckpt_path = ckpt_dir / f"swae_sub{SUBJECT_INDEX}_{tag}.pt"

torch.save({"encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "prior":   pri.state_dict(),
            "params":  {
                "subject": SUBJECT_INDEX,
                "control_sanity": CONTROL_SANITY_CHECK,
                "hand":     ("left" if USE_LEFT_HAND else "right"),
                "n_perm":   N_PERM,
                "epochs":   EPOCHS,
                "λ":        LAMBDA_SWD,
                "z_dim":    Z_DIM},
            "cluster_test": {
                "table": contingency,
                "p_value": p_value},
            "loss": hist}, ckpt_path)

print("\n Training complete — checkpoint saved to", ckpt_path)
# ===========================================================================

# Compute latent representations for each class
encoder.eval()
latent_all = []
labels_all = []
for lbl in class_labels:
    ds_lbl = CubeDataset(cubes_per_label[lbl], zscore=True)
    dl_lbl = torch.utils.data.DataLoader(ds_lbl, batch_size=256, shuffle=False)
    with torch.no_grad():
        for xb in dl_lbl:
            xb = xb.to(DEVICE)
            mean, _ = encoder(xb)
            latent_all.append(mean.cpu().numpy())
            labels_all.extend([lbl] * mean.size(0))
latent_all = np.concatenate(latent_all, axis=0)

# Perform PCA to get the first two principal components
pca = PCA(n_components=2)
pc = pca.fit_transform(latent_all)

# Plot the PCs colored by label
plt.figure(figsize=(8, 6))
unique_labels = np.unique(labels_all)
for lbl in unique_labels:
    indices = [i for i, l in enumerate(labels_all) if l == lbl]
    plt.scatter(pc[indices, 0], pc[indices, 1], label=lbl, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Latent Space Projection: First Two Principal Components")
plt.legend()
plt.tight_layout()
plt.show()