import os
import sys

# ensure parent directory is on PYTHONPATH so Classifier and Preprocessing resolve
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path  = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import numpy as np
import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# your model + training utilities
from Classifier.AE_models import MixtureVAE, train_mixture_vae, create_sliding_windows

# your preprocessing loader
from Preprocessing import get_group_epochs_subtracting_short

# reproducibility
seed_ = 42
random.seed(seed_)
np.random.seed(seed_)
torch.manual_seed(seed_)
torch.cuda.manual_seed(seed_)

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_mixture_vae_single_subject(
        device,
        participant_idx,
        *,
        latent_dim   = 2,
        epochs_num   = 25,
        beta         = 5.0,
        means        = 1,
        logvar       = 0,
        verbose      = True):
    """
    Train a MixtureVAE on a single subject's data and return latent embeddings + accuracy.
    """

    # --- load & prepare one subject ---
    epochs = get_group_epochs_subtracting_short(tmin=-5, add_hbr=False,
                              hbr_multiplier=5.0, hbr_shift=1.0)
    labels = [e.events[:, -1].astype(np.int64) - 1 for e in epochs]
    labels = labels[participant_idx]
    X = epochs[participant_idx].copy().crop(tmin=-4.5, tmax=14).get_data()

    # normalize per-channel
    n_ep, n_ch, n_t = X.shape
    for ch in range(n_ch):
        m, s = X[:, ch, :].mean(), X[:, ch, :].std()
        X[:, ch, :] = (X[:, ch, :] - m) / s

    # sliding windows → (N, C, Twin)
    X_win, lbl_win = create_sliding_windows([X], [labels], 32)
    X_t = torch.from_numpy(X_win)  # float32
    labels_bin = [1 if x >= 1 else 0 for x in lbl_win]

    loader = DataLoader(TensorDataset(X_t), batch_size=64, shuffle=True)

    # --- model & prior ---
    model = MixtureVAE(n_ch, X_t.shape[2], latent_dim).to(device)
    prior_means   = torch.ones((2, latent_dim), device=device) \
                    / np.sqrt(latent_dim) * means
    prior_means[0] *= -1
    prior_logvars = torch.ones_like(prior_means) / np.sqrt(latent_dim) * logvar
    pi_mix        = torch.tensor([0.5, 0.5], device=device)

    # --- train ---
    hist = train_mixture_vae(
        model, loader, loader,            # using same for val
        prior_means, prior_logvars, pi_mix,
        epochs_num, device,
        beta=beta, verbose=verbose
    )

    # --- evaluate & return ---
    model.eval()
    with torch.no_grad():
        mu, _   = model.encode(X_t.to(device))
        latent  = mu.cpu().numpy()

    # cluster by nearest prior mean
    d0 = ((latent - prior_means[0].cpu().numpy())**2).sum(1)
    d1 = ((latent - prior_means[1].cpu().numpy())**2).sum(1)
    pred = np.where(d0 < d1, 0, 1)

    # balanced accuracy
    acc = np.mean(np.abs(pred - np.array(labels_bin)))
    acc = (1 - acc) if acc < 0.5 else acc

    return {
        "latent":      latent,
        "pred_labels": pred,
        "labels":      labels_bin,
        "prior_means": prior_means.cpu().numpy(),
        "history":     hist,
        "acc":         acc,
    }

if __name__ == "__main__":
    # example usage:
    latent_dim=2
    results = run_mixture_vae_single_subject(
        device,
        participant_idx=4,
        latent_dim=latent_dim,
        epochs_num=40,
        beta=0.01,
        means=1,
        logvar=-1.8,
        verbose=True
    )

    acc = results["acc"]
    latent = results["latent"]
    labels = results["labels"]
    priors = results["prior_means"]

    print(f"Accuracy: {acc:.3f}")
    if latent_dim == 2:
        # scatter plot of latent space with true labels
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(latent[:, 0], latent[:, 1],
                                c=labels, cmap='coolwarm', s=20, alpha=0.6)
        plt.scatter(priors[:, 0], priors[:, 1],
                    c='black', marker='X', s=100, label='Prior Means')
        plt.title(f"MixtureVAE Latent Space (acc = {acc:.2f})")
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.legend()
        plt.tight_layout()
        plt.show()
