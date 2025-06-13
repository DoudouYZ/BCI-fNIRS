"""
plot_results.py
Visualise VAE-committee outputs for five participants.

Expected files per participant i
--------------------------------
subject_results/results_subject<i>/
    ├─ subject_<i>_right_1_seeds.npz
    ├─ subject_<i>_left_1_seeds.npz
    └─ ALL_CONTROL_subject_<i>_1_seeds.npz
Each .npz must contain (keys created by save_results.py):
    consensus, times, events, event_names, interval_means,
    samples_control, samples_tap_left, samples_tap_right,
    latents (object array [Nmodels, Nwindows, d]),
    streams (float  [Nmodels, Nwindows])
"""

from pathlib import Path
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA

# ------------------------------------------------------------------ CONFIG
plot_dir_path = "VAE_plots_with_mmd_rampup"
PLOTS_DIR   = Path(plot_dir_path);  PLOTS_DIR.mkdir(exist_ok=True)
PARTICIPANTS = [0, 1, 2, 3, 4]
SEEDS_NR    = 1
SAMPLE_SIZE = 100
N_SIM       = 10_000
RNG         = np.random.default_rng(42)
# --------------------------------------------------------------------------

def load_npz(i: int, kind: str):
    """kind ∈ {'right','left','ctrl'}"""
    if kind == "ctrl":
        fname = f"ALL_CONTROL_subject_{i}_{SEEDS_NR}_seeds.npz"
    else:
        fname = f"subject_{i}_{kind}_{SEEDS_NR}_seeds.npz"
    return np.load(Path(f"subject_results/results_subject{i}") / fname,
                   allow_pickle=True)

# ───────────────────────── helper for coloured backgrounds ───────────────
def bg(ax, xlim, colour, alpha=.15):
    ax.add_patch(Rectangle((xlim[0], ax.get_ylim()[0]),
                           xlim[1]-xlim[0], ax.get_ylim()[1]-ax.get_ylim()[0],
                           facecolor=colour, alpha=alpha, zorder=-1))


#########################################################
################## PLOTING FUNCTIONS ####################
#########################################################

# ─────────────────────────── 1. interval-mean grid ───────────────────────
def plot_interval_mean_grid():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)

    col = {"TR":"#e0ffe0", "TL":"#e0e0ff", "CTRL":"#ffe0e0"}
    order = {"Tapping_Right":0, "Tapping_Left":1, "Control":2}

    for p, ax in zip(PARTICIPANTS, axes):
        epochs = []
        for kind in ("right", "left"):
            D = load_npz(p, kind)
            for (s,e), m, lbl in zip(zip(D["events"][:-1], D["events"][1:]),
                                     D["interval_means"], D["event_names"][:-1]):
                mask = (D["times"] >= s) & (D["times"] < e)
                idx  = np.where(mask)[0]
                if idx.size: epochs.append((order[lbl.split()[0]], lbl, idx, m))
        epochs.sort(key=lambda x: x[0])

        x, y, cur = [], [], 0
        for _, lbl, idx, m in epochs:
            L = idx.size
            x.extend(range(cur, cur+L));  y.extend([m]*L)
            cat = "TR" if "Right" in lbl else ("TL" if "Left" in lbl else "CTRL")
            bg(ax, (cur, cur+L), col[cat])
            cur += L
        ax.plot(x, y, lw=1); ax.set_title(f"Subj {p}"); ax.set_xlabel("sample")
    axes[0].set_ylabel("Avg pred");  plt.tight_layout()
    plt.savefig(PLOTS_DIR/"interval_mean_grid.png", dpi=300); plt.close()

# ─────────────────────────── 2. histogram grid ───────────────────────────
def plot_histogram_grid():
    fig, axes = plt.subplots(5, 3, figsize=(14, 12),
                             sharex=True, sharey='row')

    def hist(ax, data, *, color, label, alpha, outline=False):
        if len(data) == 0:
            return
        style = dict(bins=10,
                     weights=np.ones_like(data) / len(data),
                     color=color,
                     alpha=alpha,
                     label=label)
        if outline:                     # thin outline for the “step” look
            style.update(histtype='step', linewidth=1.6)
        ax.hist(data, **style)

    for p, (axR, axL, axC) in zip(PARTICIPANTS, axes):
        D_R, D_L, D_C = (load_npz(p, k) for k in ("right", "left", "ctrl"))

        # ---------- Right vs Control ----------
        hist(axR, D_R["samples_tap_right"], color="#d62728", label="Tap-R", alpha=.35)
        hist(axR, D_R["samples_control"],   color="#1f77b4", label="Control", alpha=.35)          # blue outline, no black edges
        axR.set_title(f"Subj {p} Right vs Ctrl"); axR.set_xlim(0, 1)

        # ---------- Left vs Control -----------
        hist(axL, D_L["samples_tap_left"],  color="#ff7f0e", label="Tap-L", alpha=.35)
        hist(axL, D_L["samples_control"],   color="#1f77b4", label="Control", alpha=.85)
        axL.set_title("Left vs Ctrl"); axL.set_xlim(0, 1)

        # ---------- Control-only --------------
        hist(axC, D_C["samples_control"],   color="#1f77b4", label="Control", alpha=.85)
        axC.set_title("Ctrl-only"); axC.set_xlim(0, 1)

    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("Pred label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "histogram_grid.png", dpi=300)
    plt.close()

# ─────────────────────────── 3. Monte-Carlo grid ────────────────────────
def plot_mc_grid():
    fig, axes = plt.subplots(5,3, figsize=(14,12), sharex='col')
    for p in PARTICIPANTS:
        for j,kind in enumerate(("right","left","ctrl")):
            D = load_npz(p, kind); ax = axes[p,j]
            for lab, arr, c in [("Ctrl", D["samples_control"], 'steelblue'),
                                ("L", D.get("samples_tap_left", []),'orange'),
                                ("R", D.get("samples_tap_right",[]),'red')]:
                arr=np.asarray(arr);  n=arr.size
                if n<2: continue
                means = RNG.choice(arr, size=(N_SIM,SAMPLE_SIZE), replace=True).mean(1)
                ax.hist(means, bins=60, density=True, alpha=.4, color=c, label=lab)
            ax.set_title(f"S{p} {kind}")
    axes[-1,1].set_xlabel(f"Mean of {SAMPLE_SIZE} samples")
    axes[0,0].legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR/"mc_sampling_grid.png", dpi=300); plt.close()

# ─────────────────────────── 4. epoch-mean boxplots ─────────────────────
def plot_epoch_mean_grid():
    fig, axes = plt.subplots(5,3, figsize=(14,12), sharey=True)

    for p in PARTICIPANTS:
        for j,kind in enumerate(("right","left","ctrl")):
            D = load_npz(p, kind)
            ctl, tl, tr = [], [], []
            for (s,e), m, lbl in zip(zip(D["events"][:-1], D["events"][1:]),
                                     D["interval_means"], D["event_names"][:-1]):
                (ctl if "Control" in lbl else tl if "Left" in lbl else tr).append(m)
            data = [ctl, tl, tr] if kind!="ctrl" else [ctl]
            axes[p,j].boxplot(data, tick_labels=["Ctl","L","R"][:len(data)])
            axes[p,j].set_title(f"S{p} {kind}")
    axes[0,0].set_ylabel("Per-epoch mean")
    plt.tight_layout(); plt.savefig(PLOTS_DIR/"epoch_means.png", dpi=300); plt.close()

# ─────────────────────────── 5. latent PCA scatter ──────────────────────
def plot_latent_pca(subj_idx=0):
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    for ax, kind in zip(axes, ("left","right")):
        D = load_npz(subj_idx, kind)
        Z       = np.asarray(D["latents"][0])  # (n_windows, d)
        starts  = np.asarray(D["starts"])      # (n_windows,)
        events  = np.sort(D["events"])         # interval boundaries
        names   = D["event_names"]             # len(events)-1 labels

        # 1) compute true label for each window
        true_labels = []
        for s in starts:
            # find the interval that s falls into
            idx = np.searchsorted(events, s, side="right") - 1
            lbl = names[idx]
            if "Control" in lbl:
                true_labels.append("ctrl")
            elif "Right" in lbl:
                true_labels.append("right")
            else:
                true_labels.append("left")

        # 2) PCA
        pca = PCA(2)
        Z2  = pca.fit_transform(Z)

        # 3) map to colours
        cmap = {"ctrl":"#1f77b4",  # blue
                "right":"#d62728", # red
                "left":"#ff7f0e"}  # orange
        cols = [cmap[l] for l in true_labels]

        ax.scatter(Z2[:,0], Z2[:,1], c=cols, s=4, alpha=0.6)
        ax.set_title(f"Subj {subj_idx}  {kind.capitalize()} vs Ctrl")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR/f"latent_PCA_{subj_idx}.png", dpi=300)
    plt.close()
    

# ──────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    plot_interval_mean_grid()
    plot_histogram_grid()
    plot_mc_grid()
    plot_epoch_mean_grid()
    plot_latent_pca(subj_idx=2)
    print("✓  All plots saved to", PLOTS_DIR)
