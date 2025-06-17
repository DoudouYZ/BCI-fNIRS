import sys
import torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import mne

# ─────────────  USER SETTINGS  ─────────────
CHECKPOINT_PATH   = "checkpoints/swae_sub0_right.pt"   # ← change as needed
SAVE_FIG_PATH     = "latent_scatter.png"                  # None = don’t save
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_EVAL   = 256
N_PERM            = 1000   # number of permutations for the permutation test
MAX_PERM          = 500
# Settings for Monte Carlo sampling similar to VAE_results_plotting.py
SAMPLE_SIZE = 100
N_SIM = 10_000
RNG = np.random.default_rng(42)
# ─────────────────────────────────────────────

from WAE_preprocessing  import preprocess_for_wae_cube
from WAE_model          import CubeEncoder, CubeDecoder, GMMPrior, sliced_wasserstein, CubeDataset
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing      import get_raw_subject_data
# ─────────────────────────────────────────────

def cluster_assignment(z, prior):
    """Assign each latent vector to the nearer GMM mode (Mahalanobis)."""
    mu  = prior.mu.detach()                    # (2, z_dim)
    var = torch.exp(prior.logvar.detach())     # (2, z_dim)
    diff = z.unsqueeze(1) - mu                 # (N,2,D)
    d2   = (diff**2 / var.unsqueeze(0)).sum(dim=2)
    return torch.argmin(d2, dim=1)             # (N,)

def permutation_pvalue(binary_assignments, group_labels, n_perm=1000):
    """
    Compute a permutation-based p-value from binary assignments.
    binary_assignments: 1D numpy array of 0’s/1’s (1 if assigned to reference cluster, here 0).
    group_labels: 1D numpy array with values 0 or 1 indicating class membership corresponding to binary_assignments.
    The test statistic is the absolute difference in the fraction of ones between groups.
    """
    group_labels = np.asarray(group_labels)
    binary_assignments = np.asarray(binary_assignments)
    obs_diff = abs(binary_assignments[group_labels==0].mean() - 
                   binary_assignments[group_labels==1].mean())
    count_extreme = 0
    for _ in range(n_perm):
        permuted = np.random.permutation(binary_assignments)
        diff = abs(permuted[group_labels==0].mean() - 
                   permuted[group_labels==1].mean())
        if diff >= obs_diff:
            count_extreme += 1
    return count_extreme / n_perm

# --------------------------------------------------------------------------
# 1. Load checkpoint  -------------------------------------------------------
# --------------------------------------------------------------------------
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
params = ckpt["params"]
print("Loaded checkpoint from", CHECKPOINT_PATH)
print("Training params:", params)

SUBJECT_INDEX        = params["subject"]
CONTROL_SANITY_CHECK = params.get("control_sanity", False)
USE_LEFT_HAND        = (params["hand"] == "left")
N_PERM_MODEL         = params["n_perm"]

# --------------------------------------------------------------------------
# 2. Recreate epochs → cubes  ----------------------------------------------
# --------------------------------------------------------------------------
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


cubes_per_label = preprocess_for_wae_cube(
    epochs_dict,
    class_labels=class_labels,
    n_perm=N_PERM_MODEL,
    max_perm=MAX_PERM,        # use full set for evaluation
    rng=0,
)
print("Cubes per label:", {k: len(v) for k, v in cubes_per_label.items()})

# --------------------------------------------------------------------------
# 3. Build datasets, load model weights  -----------------------------------
# --------------------------------------------------------------------------
# Build one dataset just to know (D,C,T)
sample_cube = cubes_per_label[class_labels[0]][0]
D, C, T = sample_cube.shape
enc = CubeEncoder(D, C, z_dim=params["z_dim"]).to(DEVICE)
dec = CubeDecoder(D, C, T, z_dim=params["z_dim"]).to(DEVICE)
pri = GMMPrior(params["z_dim"]).to(DEVICE)

enc.load_state_dict(ckpt["encoder"]); enc.eval()
dec.load_state_dict(ckpt["decoder"]); dec.eval()
pri.load_state_dict(ckpt["prior"])

# --------------------------------------------------------------------------
# 4. Collect latents, build contingency table and compute permutation p-value
# --------------------------------------------------------------------------
latents, labels_text = [], []
binary_assignments = []  # will be 1 if assigned to cluster 0, else 0
group_membership = []    # 0 for class_labels[0], 1 for class_labels[1]
contingency = np.zeros((2, 2), dtype=int)

for grp, lbl in enumerate(class_labels):
    ds_lbl = CubeDataset(cubes_per_label[lbl], zscore=True)
    dl_lbl = DataLoader(ds_lbl, batch_size=BATCH_SIZE_EVAL, shuffle=False)
    with torch.no_grad():
        for xb in dl_lbl:
            xb = xb.to(DEVICE)
            μ, _ = enc(xb)
            cl = cluster_assignment(μ, pri).cpu().numpy()     # returns 0 or 1
            # Build contingency table
            contingency[grp] += np.bincount(cl, minlength=2)
            # Accumulate latent vectors
            latents.append(μ.cpu())
            # Record group membership and binary assignment (we use cluster==0 as reference)
            binary_assignments.extend((cl==0).astype(int))
            group_membership.extend([grp] * len(cl))
            labels_text.extend([lbl] * μ.shape[0])

latents = torch.cat(latents).numpy()   # (N, z_dim)
print("\nContingency (rows=labels, cols=clusters):\n", contingency)

# Compute permutation-based p-value
p_val_perm = permutation_pvalue(binary_assignments, group_membership, n_perm=N_PERM)
print(f"Permutation-based p-value = {p_val_perm:.4e}")

# --------------------------------------------------------------------------
# 5. Plot first two PCs  ----------------------------------------------------
# --------------------------------------------------------------------------
# Compute 5 principal components
print("PCA on latents...")
pcs = PCA(n_components=2).fit_transform(latents)

color_map = {
    class_labels[0]: "tab:blue",
    class_labels[1]: "tab:orange"
}
colors = [color_map[lab] for lab in labels_text]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, alpha=0.75, edgecolor="k", linewidth=0.3)
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title(f"Scatter Plot of First Two Principal Components (p = {p_val_perm:.2g})")

if SAVE_FIG_PATH is not None:
    fig.savefig(SAVE_FIG_PATH, dpi=300, bbox_inches="tight")
    print("Figure saved to", SAVE_FIG_PATH)

plt.show()
# ==========================================================================

# ────────────────────────── Monte Carlo Bootstrap Distribution ────────────
fig, ax = plt.subplots(figsize=(8,6))
for grp, label in enumerate(class_labels):
    # extract binary assignments for the current group
    group_arr = np.array(binary_assignments)[np.array(group_membership)==grp]
    if group_arr.size < 2:
        continue
    boot_means = RNG.choice(group_arr, size=(N_SIM, SAMPLE_SIZE), replace=True).mean(axis=1)
    ax.hist(boot_means, bins=60, density=True, alpha=0.4, label=label)
ax.set_xlabel("Mean cluster 0 assignment")
ax.set_ylabel("Density")
ax.set_title("Monte Carlo Bootstrap Distribution\nof Mean Cluster Assignment")
ax.legend()
plt.show()

# Separate binary assignments into two groups
group0 = np.array(binary_assignments)[np.array(group_membership) == 0]
group1 = np.array(binary_assignments)[np.array(group_membership) == 1]

# Calculate the observed absolute difference in means
obs_diff = abs(group0.mean() - group1.mean())

# Monte Carlo Bootstrapping under the null hypothesis that both groups come from the same distribution
combined = np.concatenate([group0, group1])
n0, n1 = len(group0), len(group1)
N_BOOT = 10_000
extreme_count = 0

for _ in range(N_BOOT):
    permuted = RNG.permutation(combined)
    boot0 = permuted[:n0]
    boot1 = permuted[n0:]
    diff = abs(boot0.mean() - boot1.mean())
    if diff >= obs_diff:
        extreme_count += 1

boot_p_value = extreme_count / N_BOOT
print("Monte Carlo bootstrap p-value = {:.4e}".format(boot_p_value))