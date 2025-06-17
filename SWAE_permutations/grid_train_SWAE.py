# =========================  grid_train_wae.py  ===========================
"""
Grid-search runner for the SWAE finger-tapping experiment.

• Loops across   SEED × SUBJECT × HAND × SANITY × N_PERM × LAMBDA_SWD × Z_DIM
• After every run it …  
    1. writes/updates one CSV per seed with the p-value  
    2. stores a checkpoint named with all hyper-parameters  
• Safe to interrupt — already-computed cells are skipped next launch.
"""
import sys, os, math, itertools, json
import torch, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import fisher_exact
from torch.utils.data import DataLoader

# ────────────────────────────────────────────────────────────────────────────
#  CONSTANT PARTS (rarely changed)
# ────────────────────────────────────────────────────────────────────────────
MAX_PERM   = 1_000
BATCH_SIZE = 32
EPOCHS     = 175
LR         = 1e-3
N_PROJ     = 150
VERBOSE    = True
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

#  util imports
from WAE_preprocessing  import preprocess_for_wae_cube
from WAE_model          import CubeEncoder, CubeDecoder, GMMPrior, sliced_wasserstein, CubeDataset
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing      import get_raw_subject_data

# ────────────────────────────────────────────────────────────────────────────
#  SEARCH SPACE
# ────────────────────────────────────────────────────────────────────────────
SEEDS            = range(3)
SUBJECTS         = range(5)
LEFT_HAND        = (True, False)
CONTROL_ONLY     = (False, True)
N_PERMS          = (3, 4, 5, 7, 10)
LAMBDA_SWDS      = (15, 25, 50)
Z_DIMS           = (4, 6, 8)

# column key helper (now includes z_dim)
def col_key(z_dim, hand, sanity, n_perm, lam):
    return f"Z{z_dim}_H{int(hand)}_S{int(sanity)}_P{n_perm}_L{lam}"

# ────────────────────────────────────────────────────────────────────────────
#  CLUSTER ASSIGNMENT HELPERS
# ────────────────────────────────────────────────────────────────────────────
def cluster_assignment(z, prior):
    mu  = prior.mu.detach()
    var = torch.exp(prior.logvar.detach())
    d2  = ((z.unsqueeze(1) - mu)**2 / var.unsqueeze(0)).sum(dim=2)
    return torch.argmin(d2, dim=1)

def fisher_pvalue(enc, pri, cubes_per_label, class_labels, device):
    contingency = np.zeros((2, 2), dtype=int)
    enc.eval()
    for row, lbl in enumerate(class_labels):
        ds = CubeDataset(cubes_per_label[lbl], zscore=True)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        with torch.no_grad():
            for xb in dl:
                xb = xb.to(device)
                μ, _ = enc(xb)
                cl   = cluster_assignment(μ, pri).cpu().numpy()
                for c in cl: 
                    contingency[row, c] += 1
    _, p_val = fisher_exact(contingency)
    return float(p_val), contingency

def permutation_pvalue(enc, pri, cubes_per_label, class_labels, device, n_perm=1000):
    # Compute observed contingency and a statistic (here: difference in cluster0 frequencies)
    obs_counts = np.zeros(2, dtype=int)  # counts for cluster 0 for each class
    totals = np.zeros(2, dtype=int)
    enc.eval()
    for row, lbl in enumerate(class_labels):
        ds = CubeDataset(cubes_per_label[lbl], zscore=True)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        with torch.no_grad():
            for xb in dl:
                xb = xb.to(device)
                μ, _ = enc(xb)
                clusters = cluster_assignment(μ, pri).cpu().numpy()
                obs_counts[row] += (clusters == 0).sum()
                totals[row] += len(clusters)
    obs_stat = abs(obs_counts[0]/totals[0] - obs_counts[1]/totals[1])
    
    # Collect all cluster0 assignments from both classes
    all_assignments = []
    for row, lbl in enumerate(class_labels):
        ds = CubeDataset(cubes_per_label[lbl], zscore=True)
        dl = DataLoader(ds, batch_size=256, shuffle=False)
        with torch.no_grad():
            for xb in dl:
                xb = xb.to(device)
                μ, _ = enc(xb)
                clusters = cluster_assignment(μ, pri).cpu().numpy()
                # record whether each sample was assigned to cluster 0 (1) or not (0)
                all_assignments.extend((clusters==0).astype(int))
    all_assignments = np.array(all_assignments)
    n_total = all_assignments.shape[0]
    n_class0 = totals[0]
    
    # Permutation: randomly split all assignments into two groups of sizes totals[0] and totals[1]
    count_extreme = 0
    for _ in range(n_perm):
        permuted = np.random.permutation(all_assignments)
        group0 = permuted[:n_class0]
        group1 = permuted[n_class0:]
        stat = abs(np.mean(group0) - np.mean(group1))
        if stat >= obs_stat:
            count_extreme += 1
    p_val = count_extreme / n_perm
    # Also return the observed contingency
    contingency = np.array([[obs_counts[0], totals[0]-obs_counts[0]],
                            [obs_counts[1], totals[1]-obs_counts[1]]])
    return p_val, contingency

# ────────────────────────────────────────────────────────────────────────────
#  MAIN TRAIN-AND-LOG FUNCTION
# ────────────────────────────────────────────────────────────────────────────
def run_one(seed, subject, hand, control_only, n_perm, lam_swd, z_dim,
            csv_path: Path, checkpoints_dir: Path):

    # --- skip if already done ------------------------------------------------
    key = col_key(z_dim, hand, control_only, n_perm, lam_swd)
    df  = pd.read_csv(csv_path, index_col=0)
    if not math.isnan(df.loc[subject, key]):
        print(f"Seed{seed} subj{subject} {key}  …already done")
        return                                    # nothing to do

    # --- reproducibility -----------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- load epochs ---------------------------------------------------------
    full_epochs = get_raw_subject_data(subject)
    if control_only:
        ctl_all = full_epochs["Control"]
        idx     = np.random.permutation(len(ctl_all))
        split   = len(idx)//2
        epochs_dict = {"Control_A": ctl_all[idx[:split]],
                       "Control_B": ctl_all[idx[split:]]}
        class_labels = ["Control_A", "Control_B"]
    else:
        tap_lbl = "Tapping_Left" if hand else "Tapping_Right"
        epochs_dict  = {tap_lbl:  full_epochs[tap_lbl],
                        "Control": full_epochs["Control"]}
        class_labels = [tap_lbl, "Control"]

    print("Epoch counts:",
          {lbl: len(ep) for lbl, ep in epochs_dict.items()})

    # --- build cubes ---------------------------------------------------------
    cubes_per_label = preprocess_for_wae_cube(
        epochs_dict,
        class_labels=class_labels,
        n_perm=n_perm,
        max_perm=MAX_PERM,
        rng=seed,
    )
    for k, v in cubes_per_label.items():
        print(f"   {len(v):5d} cubes from {k}")

    cubes_all = cubes_per_label[class_labels[0]] + cubes_per_label[class_labels[1]]
    ds = CubeDataset(cubes_all, zscore=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Cube shape: (n_perm={ds.D}, C={ds.C}, T={ds.T})")

    # --- init model ----------------------------------------------------------
    enc = CubeEncoder(ds.D, ds.C, z_dim=z_dim).to(DEVICE)
    dec = CubeDecoder(ds.D, ds.C, ds.T, z_dim=z_dim).to(DEVICE)
    pri = GMMPrior(z_dim).to(DEVICE)
    opt = torch.optim.Adam(
        list(enc.parameters())+list(dec.parameters())+list(pri.parameters()), lr=LR)
    mse = torch.nn.MSELoss()

    # --- train ---------------------------------------------------------------
    print(f"Training with hyper-params:")
    print(f"seed={seed}, subject={subject}, z_dim={z_dim}, hand={'Left' if hand else 'Right'}, control_only={control_only}, "
          f"n_perm={n_perm}, lambda={lam_swd}")
    for ep in range(1, EPOCHS+1):
        rec_sum = swd_sum = 0.0
        for x in dl:
            x = x.to(DEVICE)
            mean, log_sigma_2 = enc(x)
            # z = mean + torch.exp(0.5 * log_sigma_2) * torch.randn_like(mean)
            clamped_log_sigma2 = torch.clamp(log_sigma_2, min=-12, max=12)
            z = mean + torch.exp(0.5 * clamped_log_sigma2) * torch.randn_like(mean)
            x_hat = dec(z)
            l_rec = mse(x_hat, x)
            l_swd = sliced_wasserstein(z, pri.sample(x.size(0)), n_proj=N_PROJ)
            loss  = l_rec + lam_swd * l_swd
            opt.zero_grad()
            loss.backward()
            opt.step()
            rec_sum += l_rec.item()
            swd_sum += l_swd.item()
        if VERBOSE:
            avg_rec = rec_sum / len(dl)
            avg_swd = swd_sum / len(dl)
            if ep % 10 == 0 or ep == EPOCHS:
                print(f"Epoch {ep:3d} | recon {avg_rec:.5f} | swd {avg_swd:.5f}")

    # --- evaluate p-value ----------------------------------------------------
    # p_val, cont = fisher_pvalue(enc, pri, cubes_per_label, class_labels, DEVICE)
    p_val, cont = permutation_pvalue(enc, pri, cubes_per_label, class_labels, DEVICE, n_perm=1000)
    print(f"Seed{seed} subj{subject} {key}  p={p_val:.3e}")

    # --- write to CSV (in-place update) --------------------------------------
    df.loc[subject, key] = p_val
    df.to_csv(csv_path, float_format="%.6e")

    # --- save checkpoint -----------------------------------------------------
    ck_name = f"swae_s{seed}_sub{subject}_{key}.pt"
    torch.save({
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "prior":   pri.state_dict(),
        "params":  {"seed": seed,
                    "subject": subject,
                    "z_dim": z_dim,
                    "hand": hand,
                    "sanity": control_only,
                    "n_perm": n_perm,
                    "λ": lam_swd},
        "contingency": cont,
        "p_value": p_val}, checkpoints_dir / ck_name)

# ────────────────────────────────────────────────────────────────────────────
#  PREP CSV FILES (one per seed)
# ────────────────────────────────────────────────────────────────────────────
results_dir      = Path("grid_results")
results_dir.mkdir(exist_ok=True)
checkpoints_dir  = Path("checkpoints")
checkpoints_dir.mkdir(exist_ok=True)

COLS = [col_key(z, h, s, p, l) 
        for (z, h, s, p, l) in itertools.product(Z_DIMS, LEFT_HAND, CONTROL_ONLY, N_PERMS, LAMBDA_SWDS)]

for seed in SEEDS:
    csv_path = results_dir / f"pvals_seed{seed}.csv"
    if csv_path.exists():
        # ensure all expected columns exist (in case you extended the grid)
        df = pd.read_csv(csv_path, index_col=0)
        missing = [c for c in COLS if c not in df.columns]
        for m in missing: 
            df[m] = np.nan
    else:  # create fresh file
        df = pd.DataFrame(np.nan, index=list(SUBJECTS), columns=COLS)
    df.to_csv(csv_path)  # save (or overwrite) header

# ────────────────────────────────────────────────────────────────────────────
#  GRID LOOP  (nested but short-circuit skips)
# ────────────────────────────────────────────────────────────────────────────
for seed in SEEDS:
    csv_path = results_dir / f"pvals_seed{seed}.csv"
    for (subject, z_dim, hand, sanity, n_perm, lam) in itertools.product(
            SUBJECTS, Z_DIMS, LEFT_HAND, CONTROL_ONLY, N_PERMS, LAMBDA_SWDS):
        # avoid meaningless combos: when sanity==True, hand choice is irrelevant
        if sanity and hand is False:
            continue
        run_one(seed, subject, hand, sanity, n_perm, lam, z_dim,
                csv_path=csv_path, checkpoints_dir=checkpoints_dir)

print("\n grid search finished.  CSV files are in", results_dir)
# ===========================================================================