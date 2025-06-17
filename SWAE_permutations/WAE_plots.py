"""
This script loads a saved SWAE checkpoint, rebuilds the dataset and model, computes a 2D latent scatter plot,
and saves the plot to the "WAE_plots" folder.
"""

import os
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

# Import necessary modules from your project
from WAE_model import CubeEncoder, CubeDataset
from WAE_preprocessing import preprocess_for_wae_cube

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from Preprocessing import get_raw_subject_data

def latent_scatter_2pc(
    encoder: torch.nn.Module,
    cubes_per_label: dict,
    batch_size: int = 32,
    device: str = "cpu",
    title: str = "Latent space – first two PCs",
):
    """
    Plot the first two principal components of the encoder-latent vectors.

    Parameters
    ----------
    encoder : trained CubeEncoder
    cubes_per_label : dict {label: list[np.ndarray]}  (output of preprocess_for_wae_cube)
    batch_size : DataLoader batch size for latent extraction
    device : "cpu" or "cuda"
    title : plot title

    Returns
    -------
    fig : matplotlib Figure instance
    """
    encoder.eval()
    encoder.to(device)

    zs, lbls = [], []
    for lbl, cubes in cubes_per_label.items():
        ds = CubeDataset(cubes, zscore=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for x in dl:
                x = x.to(device)
                mu, _ = encoder(x)        # use μ as deterministic latent
                zs.append(mu.cpu())
                lbls.extend([lbl] * mu.shape[0])

    Z = torch.cat(zs).numpy()             # (N, z_dim)
    pcs = PCA(n_components=2).fit_transform(Z)  # (N, 2)

    # colour map: Control = blue, Tapping = orange
    colors = ["tab:blue" if l.lower().startswith("control") else "tab:orange" for l in lbls]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, alpha=0.7, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    ax.grid(True)

    # build legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",  label="Control", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", label="Tapping", markersize=8),
    ]
    ax.legend(handles=handles, frameon=False)

    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot latent space from saved SWAE checkpoint.")
    # parser.add_argument("--checkpoint", type=str, default="checkpoints/swae_sub2_right.pt",
    #                     help="Path to SWAE checkpoint file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/swae_sub2_ctlsplit.pt",
                        help="Path to SWAE checkpoint file")
    parser.add_argument("--max_perm", type=int, default=5000,
                        help="Cap on number of cubes per label during preprocessing")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size for latent extraction")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load the checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint file not found: {ckpt_path}")
        return

    # checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    params = checkpoint["params"]
    subject = params["subject"]
    hand = params["hand"]  # expected "left" or "right"
    n_perm = params["n_perm"]
    z_dim = params["z_dim"]

    # Reconstruct the dataset from raw epochs
    print("→ Loading subject", subject)
    full_epochs = get_raw_subject_data(subject=subject)

    # Set label names based on hand choice
    label_tap = "Tapping_Left" if hand == "left" else "Tapping_Right"
    label_ctl = "Control"

    epochs_dict = {
        label_tap: full_epochs[label_tap],
        label_ctl: full_epochs[label_ctl],
    }
    cubes_per_label = preprocess_for_wae_cube(
        epochs_dict,
        class_labels=[label_tap, label_ctl],
        n_perm=n_perm,
        max_perm=args.max_perm,
        rng=0,
    )

    # Merge cubes to build the CubeDataset (needed for shape inference)
    cubes_all = cubes_per_label[label_tap] + cubes_per_label[label_ctl]
    ds = CubeDataset(cubes_all, zscore=True)

    # Re-create the encoder with correct dimensions
    encoder = CubeEncoder(ds.D, ds.C, z_dim=z_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()

    # Create latent space plot
    title = f"Subject {subject} – latent PCs"
    fig = latent_scatter_2pc(
        encoder=encoder,
        cubes_per_label=cubes_per_label,
        batch_size=args.batch,
        device=device,
        title=title,
    )

    # Create output folder "WAE_plots" inside current folder
    plots_dir = Path("WAE_plots")
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / "latent_space_plot.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    print("Plot saved to", plot_path)
    plt.close()

if __name__ == "__main__":
    main()