# AE_models.py
# ------------------------------------------------------------
# Auto-Encoder / VAE building blocks for the fNIRS project
# ------------------------------------------------------------
from __future__ import annotations
import math, numpy as np
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ───────────────────────── Cropping helper ───────────────────
class Cropping1D(nn.Module):
    """Crop *crop_right* last time-steps along dim=2."""
    def __init__(self, crop_right: int):
        super().__init__()
        self.crop_right = crop_right

    def forward(self, x):
        return x[:, :, :-self.crop_right] if self.crop_right else x


# ─────────────────────────── Encoder ─────────────────────────
class Encoder(nn.Module):
    """
    Five-conv, two-pool stack:

        Conv1 → Conv2 → AvgPool
        Conv3 → Conv4 → AvgPool
        Conv5 → flatten → FC → latent

    `k_size` and `out_channels` are 5-element tuples so you can
    experiment from outside.
    """
    def __init__(self,
                 input_channels: int,
                 input_length:   int,
                 latent_dim:     int,
                 k_size:         Tuple[int,int,int,int,int] = (9,9,7,7,3),
                 out_channels:   Tuple[int,int,int,int,int] = (16,16,32,64,32)):

        super().__init__()
        k1,k2,k3,k4,k5 = k_size
        c1,c2,c3,c4,c5 = out_channels

        # ---------------- conv stack -----------------
        self.conv1 = nn.Conv1d(input_channels, c1, k1, padding=k1//2)
        self.conv2 = nn.Conv1d(c1, c2, k2, padding=k2//2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(c2, c3, k3, padding=k3//2)
        self.conv4 = nn.Conv1d(c3, c4, k4, padding=k4//2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(c4, c5, k5, padding=k5//2)

        # -------------- output length ----------------
        def L_out(L, k, s=1, p=0):             # formula once
            return (L + 2*p - k)//s + 1

        L = input_length
        L = L_out(L, k1, 1, k1//2)
        L = L_out(L, k2, 1, k2//2)
        L = L_out(L, 2,  2, 0)          # pool
        L = L_out(L, k3, 1, k3//2)
        L = L_out(L, k4, 1, k4//2)
        L = L_out(L, 2,  2, 0)          # pool
        L = L_out(L, k5, 1, k5//2)

        self.conv_output_length = L     # save for decoder

        flat = c5 * L
        self.fc1 = nn.Linear(flat, flat)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(flat, latent_dim)

    # ----------------------------------------------------------
    def forward(self, x):               # (B, C, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))

        x = x.flatten(1)                # → (B, flat)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)              # latent



# ─────────────────────────── Decoder ─────────────────────────
class MirrorDecoder(nn.Module):
    """
    Perfect mirror of the above 5-conv Encoder.
    Two up-samples (nearest) undo the two AvgPools.
    """
    def __init__(self,
                 input_channels: int,
                 input_length:   int,
                 latent_dim:     int,
                 k_size:         Sequence[int],
                 out_channels:   Sequence[int],
                 L_enc:          int):
        super().__init__()

        k1,k2,k3,k4,k5 = k_size
        c1,c2,c3,c4,c5 = out_channels
        flat           = c5 * L_enc

        layers = [
            # (1) FC → unflatten
            nn.Linear(latent_dim, flat),
            nn.Unflatten(dim=1, unflattened_size=(c5, L_enc)),

            # (2) mirror of conv5 + pool2
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(c5, c4, k5, padding=k5//2),
            nn.ReLU(inplace=True),

            # (3) mirror of conv4 (still in up-sampled zone)
            nn.Conv1d(c4, c3, k4, padding=k4//2),
            nn.ReLU(inplace=True),

            # (4) mirror of conv3 + pool1
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(c3, c2, k3, padding=k3//2),
            nn.ReLU(inplace=True),

            # (5) mirror of conv2
            nn.Conv1d(c2, c1, k2, padding=k2//2),
            nn.ReLU(inplace=True),

            # (6) mirror of conv1
            nn.Conv1d(c1, input_channels, k1, padding=k1//2)
        ]

        # we did two 2× up-samples → output length L_enc × 4
        out_len = L_enc * 4
        if out_len > input_length:
            layers.append(Cropping1D(out_len - input_length))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)



# ───────────────────── Reconstruction Autoencoder ────────────
class ReconstructionAutoencoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 input_length:   int,
                 latent_dim:     int,
                 k_size=(9,9,7,7,3),
                 out_channels=(16,16,32,64,32)):
        super().__init__()
        self.encoder = Encoder(input_channels, input_length, latent_dim,
                               k_size=k_size, out_channels=out_channels)
        self.decoder = MirrorDecoder(input_channels, input_length, latent_dim,
                                     k_size, out_channels,
                                     self.encoder.conv_output_length)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_latent(self, x):
        return self.encoder(x)


# ───────────────────── Classification Autoencoder ────────────
class ClassificationAutoencoder(nn.Module):
    def __init__(self, input_channels, input_length,
                 latent_dim, num_classes,
                 k_size=(9,9,7,7,3),
                 out_channels=(16,16,32,64,32)):
        super().__init__()
        self.encoder = Encoder(input_channels, input_length, latent_dim,
                               k_size=k_size, out_channels=out_channels)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.head(self.encoder(x))

    def get_latent(self, x):
        return self.encoder(x)


# ───────────────────────── Mixture-prior VAE ─────────────────
class MixtureVAE(nn.Module):
    def __init__(self,
                 input_channels, input_length, latent_dim,
                 k_size=(9,9,7,7,3),
                 out_channels=(16,16,32,64,32)):
        super().__init__()
        self.encoder_core = Encoder(input_channels, input_length, latent_dim,
                                    k_size=k_size, out_channels=out_channels)
        hidden_sz = self.encoder_core.fc2.in_features    # before latent

        # replace encoder’s last head with μ & logσ² heads
        self.encoder_core.fc2 = nn.Identity()
        self.mu_head     = nn.Linear(hidden_sz, latent_dim)
        self.logvar_head = nn.Linear(hidden_sz, latent_dim)

        # mirror decoder
        self.decoder = MirrorDecoder(input_channels, input_length, latent_dim,
                                     k_size, out_channels,
                                     self.encoder_core.conv_output_length)

    # ------------------------ encode / forward -------------------------
    def encode(self, x):
        h = self.encoder_core(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar


# ──────────────────────── util: sliding windows ──────────────
def create_sliding_windows_no_classes(data: np.ndarray,
                                      window_length: int,
                                      times: np.ndarray = None,
                                      events: np.ndarray = None,
                                      buffer: float = 0.0):
    """
    (Same helper the committee uses.)

    If *events* is given we discard any window whose centre lies within
    ±buffer seconds of an event boundary.
    Returns
    -------
    windows : (N, C, L)  float32
    starts  : (N,)       float – start-time (s) of every window
    """
    C, T = data.shape
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=window_length, axis=1)        # (C, N, L)
    windows = windows.transpose(1,0,2)                   # (N,C,L)

    if times is None or events is None or buffer <= 0:
        starts = np.asarray(range(windows.shape[0]), dtype=float)  # dummy
        return windows.astype(np.float32), starts

    # real time of left edge of each window
    starts = times[:windows.shape[0]]
    centres = starts + window_length / 2 / (times[1]-times[0]) * (times[1]-times[0])

    keep = np.ones_like(centres, bool)
    for ev in events:
        keep &= np.abs(centres - ev) > buffer
    return windows[keep].astype(np.float32), starts[keep]


# ──────────────────────── util: label projection ─────────────
def per_timepoint_labels_sparse(window_labels: np.ndarray,
                                window_length: int,
                                starts: np.ndarray,
                                n_times: int):
    """
    Project window-level labels (stride=1) to per-sample labels.

    Returns
    -------
    sample_labels : (n_times,) float with NaN for uncovered samples
    covered_mask  : bool mask same length
    """
    y = np.full(n_times, np.nan, dtype=float)
    for idx, s in enumerate(starts.astype(int)):
        y[s:s+window_length] = np.nanmean(
            [y[s:s+window_length], np.full(window_length, window_labels[idx])],
            axis=0)
    return y, ~np.isnan(y)


# ──────────────────────── VAE loss helpers ───────────────────
LOG_2PI = math.log(2*math.pi)

def kl_mixture_gaussian(mu, logvar, prior_means, prior_logvars, pi):
    """
    One-sample MC estimate of KL(q||p_mix).  Shapes:
        mu/logvar : (B,D)     K=2 mixture components
        prior_*   : (K,D)
        pi        : (K,)
    """
    std  = torch.exp(0.5*logvar)
    z    = mu + std * torch.randn_like(std)
    log_q = (-0.5*(LOG_2PI + logvar) - 0.5*((z-mu)**2)/logvar.exp()).sum(1)

    log_p_components = []
    for k in range(prior_means.size(0)):
        log_pk = (-0.5*(LOG_2PI + prior_logvars[k])
                  -0.5*((z-prior_means[k])**2)/prior_logvars[k].exp()).sum(1)
        log_p_components.append(torch.log(pi[k]) + log_pk)
    log_p = torch.logsumexp(torch.stack(log_p_components, dim=1), dim=1)

    return (log_q - log_p).mean()


# ──────────────────────── trainer helpers ────────────────────
def train_mixture_vae(model, train_loader, val_loader,
                      prior_means, prior_logvars, pi,
                      epochs, device, beta=1.0, verbose=False):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    hist = {'tr_rec':[], 'tr_kl':[], 'va_rec':[], 'va_kl':[]}
    iters = tqdm(range(epochs)) if verbose else range(epochs)

    for ep in iters:
        # -------- train --------
        model.train()
        trR = trK = 0.
        for (x,) in train_loader:
            x = x.to(device)
            recon, mu, lv = model(x)
            rec = mse(recon, x)
            kl  = kl_mixture_gaussian(mu, lv,
                                      prior_means, prior_logvars, pi)
            loss = rec + beta*kl
            opt.zero_grad(); loss.backward(); opt.step()
            trR += rec.item()*x.size(0); trK += kl.item()*x.size(0)

        # -------- val ----------
        model.eval()
        vaR = vaK = 0.
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, mu, lv = model(x)
                vaR += mse(recon, x).item()*x.size(0)
                vaK += kl_mixture_gaussian(mu, lv,
                                           prior_means, prior_logvars, pi).item()*x.size(0)

        Ntr, Nva = len(train_loader.dataset), len(val_loader.dataset)
        hist['tr_rec'].append(trR/Ntr); hist['tr_kl'].append(trK/Ntr)
        hist['va_rec'].append(vaR/Nva); hist['va_kl'].append(vaK/Nva)

        if verbose:
            print(f"E{ep+1:03}/{epochs}  rec {hist['tr_rec'][-1]:.4e}  "
                  f"kl {hist['tr_kl'][-1]:.4e}")

    return hist
