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

LOG_2PI = math.log(2 * math.pi)

def mmd_multiscale(z: torch.Tensor,
                   z_prior: torch.Tensor,
                   sigmas: Sequence[float] = (0.5, 1.0, 2.0, 4.0)
                  ) -> torch.Tensor:
    """Sum of RBF-MMD at multiple bandwidths."""
    return sum(mmd_rbf(z, z_prior, sigma=s) for s in sigmas)


def mmd_rbf(z: torch.Tensor, z_prior: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Unbiased linear‑time MMD with RBF kernel (works on any device)."""
    # pairwise squared distances; broadcasting keeps memory modest
    dist_xx = (z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(-1)
    dist_yy = (z_prior.unsqueeze(1) - z_prior.unsqueeze(0)).pow(2).sum(-1)
    dist_xy = (z.unsqueeze(1) - z_prior.unsqueeze(0)).pow(2).sum(-1)

    # clamp distances so exp() never sees inf
    dist_xx = dist_xx.clamp(max=1e12)
    dist_yy = dist_yy.clamp(max=1e12)
    dist_xy = dist_xy.clamp(max=1e12)

    k_xx = torch.exp(-dist_xx / (2 * sigma ** 2)).mean()
    k_yy = torch.exp(-dist_yy / (2 * sigma ** 2)).mean()
    k_xy = torch.exp(-dist_xy / (2 * sigma ** 2)).mean()
    return k_xx + k_yy - 2.0 * k_xy


@torch.no_grad()
def sample_mixture_prior(n: int,
                         prior_means: torch.Tensor,
                         prior_logvars: torch.Tensor,
                         pi: torch.Tensor) -> torch.Tensor:
    """Draw *n* latent samples from the diagonal Gaussian mixture prior."""
    K, D = prior_means.shape
    device = prior_means.device
    comps = torch.multinomial(pi, n, replacement=True)          # (n,)
    mu = prior_means[comps]
    std = (0.5 * prior_logvars[comps]).exp()
    return mu + std * torch.randn((n, D), device=device)



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
    
# ===================== APPEND BELOW THIS LINE =========================
# Structured encoder that respects 
#    ─ hemispheres (left 20 ch vs right 20 ch)  
#    ─ chromophores (HbO vs HbR)  
#    by keeping the four 10-channel groups separate for the first two
#    convolutions (groups = 4).  A hemispheric-difference feature is then
#    concatenated before later layers are allowed to mix everything.

class StructuredEncoder(nn.Module):
    def __init__(self,
                 input_length: int,
                 latent_dim:   int,
                 k_size=(9, 9, 7, 7, 3),
                 out_channels=(16, 16, 32, 64, 32)):
        super().__init__()
        k1, k2, k3, k4, k5 = k_size
        c1, c2, c3, c4, c5 = out_channels

        # ---- channel order: [left-HbO | right-HbO | left-HbR | right-HbR]
        left_hbo   = list(range(0, 20, 2))
        right_hbo  = list(range(20, 40, 2))
        left_hbr   = list(range(1, 20, 2))
        right_hbr  = list(range(21, 40, 2))
        self.register_buffer("reorder_idx",
                             torch.tensor(left_hbo + right_hbo +
                                          left_hbr + right_hbr,
                                          dtype=torch.long))

        # ─── grouped convs keep groups separate ───
        self.gconv1 = nn.Conv1d(40, c1, k1, padding=k1 // 2, groups=4)
        self.gconv2 = nn.Conv1d(c1, c1, k2, padding=k2 // 2, groups=4)
        self.pool1  = nn.MaxPool1d(2)                # first temporal down-sample

        # hemispheric-difference feature will be appended after pool1
        self.conv3 = nn.Conv1d(c1 + 1, c2, k3, padding=k3 // 2)
        self.conv4 = nn.Conv1d(c2, c3, k4, padding=k4 // 2)
        self.pool2 = nn.MaxPool1d(2)                # second down-sample
        self.conv5 = nn.Conv1d(c3, c4, k5, padding=k5 // 2)

        # ---- compute flattened length for FC layers ----
        def L_out(L, k, s=1, p=0):  # convenience
            return (L + 2 * p - k) // s + 1
        L = input_length
        L = L_out(L, k1, 1, k1 // 2)
        L = L_out(L, k2, 1, k2 // 2)
        L = L_out(L, 2,  2, 0)      # pool1
        L = L_out(L, k3, 1, k3 // 2)
        L = L_out(L, k4, 1, k4 // 2)
        L = L_out(L, 2,  2, 0)      # pool2
        L = L_out(L, k5, 1, k5 // 2)
        self.conv_output_length = L
        flat = c4 * L

        self.fc_h = nn.Linear(flat, flat // 2)
        self.drop = nn.Dropout(0.3)
        self.out_features = flat // 2               # for VAE heads later

    # ------------------------------------------------------------------
    def forward(self, x):                           # x: (B, 40, T)
        # reorder so grouped conv “chunks” are contiguous
        x = x[:, self.reorder_idx, :]

        # keep four groups separate
        x = F.relu(self.gconv1(x))
        x = F.relu(self.gconv2(x))
        x = self.pool1(x)

        # hemispheric difference feature (right − left, mean over HbO + HbR)
        left  = x[:, :x.size(1)//2, :]              # first ½ = left hemi
        right = x[:, x.size(1)//2:, :]              # second ½ = right hemi
        diff  = (right.mean(1, keepdim=True) -
                 left.mean(1,  keepdim=True))       # shape (B,1,T)

        x = torch.cat([x, diff], dim=1)             # now c1+1 channels

        # now allow full mixing
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))

        x = x.flatten(1)
        x = self.drop(F.relu(self.fc_h(x)))
        return x                                   # (B, out_features)

# ❷ VAE that plugs the structured encoder into the rest of your stack
class NewMixtureVAE(nn.Module):
    def __init__(self,
                 input_length: int,
                 latent_dim:   int,
                 k_size=(9, 9, 7, 7, 3),
                 out_channels=(16, 16, 32, 64, 32)):
        super().__init__()
        # encoder core delivers feature vector h
        self.encoder_core = StructuredEncoder(input_length,
                                              latent_dim,
                                              k_size=k_size,
                                              out_channels=out_channels)
        hdim = self.encoder_core.out_features
        self.mu_head     = nn.Linear(hdim, latent_dim)
        self.logvar_head = nn.Linear(hdim, latent_dim)

        # mirror decoder (one up-sample per pool in encoder)
        self.decoder = MirrorDecoder(
            input_channels=40,                    # still 40 original chans
            input_length=input_length,
            latent_dim=latent_dim,
            k_size=k_size,
            out_channels=out_channels,
            L_enc=self.encoder_core.conv_output_length)

    # --------------------------------------------------------------
    def encode(self, x):
        h = self.encoder_core(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        logvar = logvar.clamp(-10.0, 10.0)
        z = self.reparameterise(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
# ===================== END OF APPEND =================================



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
        logvar  = logvar.clamp(min=-10.0, max=10.0)   # keeps std ∈ [e^-5, e^5] ~ [0, 150]
        z          = self.reparameterise(mu, logvar)
        recon      = self.decoder(z)
        return recon, mu, logvar


# ──────────────────────── util: sliding windows ──────────────
# def create_sliding_windows_no_classes(data: np.ndarray,
#                                       window_length: int,
#                                       times: np.ndarray = None,
#                                       events: np.ndarray = None,
#                                       buffer: float = 0.0):
#     """
#     (Same helper the committee uses.)

#     If *events* is given we discard any window whose centre lies within
#     ±buffer seconds of an event boundary.
#     Returns
#     -------
#     windows : (N, C, L)  float32
#     starts  : (N,)       float – start-time (s) of every window
#     """
#     C, T = data.shape
#     windows = np.lib.stride_tricks.sliding_window_view(
#         data, window_shape=window_length, axis=1)        # (C, N, L)
#     windows = windows.transpose(1,0,2)                   # (N,C,L)

#     if times is None or events is None or buffer <= 0:
#         starts = np.asarray(range(windows.shape[0]), dtype=float)  # dummy
#         return windows.astype(np.float32), starts

#     # real time of left edge of each window
#     starts = times[:windows.shape[0]]
#     centres = starts + window_length / 2 / (times[1]-times[0]) * (times[1]-times[0])

#     keep = np.ones_like(centres, bool)
#     for ev in events:
#         keep &= np.abs(centres - ev) > buffer
#     return windows[keep].astype(np.float32), starts[keep]

def create_sliding_windows_no_classes(data: np.ndarray,
                                      window_length: int,
                                      times: np.ndarray = None,
                                      events: np.ndarray = None,
                                      buffer: float = 0.0):
    """
    (Same helper the committee uses.)

    If *events* is given we discard any window whose centre lies within
    ±buffer seconds of an event boundary.
    Additionally, we now compute an epoch id for every window based on where
    its center falls in the passed events boundaries.
    Returns
    -------
    windows : (N, C, L)  float32
    starts  : (N,)       float – start-time of every window
    epoch_ids: (N,)      int – epoch id for every window
    """
    C, T = data.shape
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=window_length, axis=1)
    windows = windows.transpose(1, 0, 2)

    if times is None or events is None or buffer <= 0:
        starts = np.arange(windows.shape[0], dtype=float)
        # dummy epoch ids (all the same)
        epoch_ids = np.zeros_like(starts, dtype=int)
        return windows.astype(np.float32), starts, epoch_ids

    # left edge times
    starts = times[:windows.shape[0]]
    # compute the window centre time. (Assumes uniform sampling.)
    dt = times[1]-times[0]
    centres = starts + window_length/2 * dt

    keep = np.ones_like(centres, bool)
    for ev in events:
        keep &= np.abs(centres - ev) > buffer
    windows = windows[keep]
    starts  = starts[keep]
    centres = centres[keep]

    # assign each centre to an epoch id from events.
    # For each centre, the epoch is the index of the last event not greater than centre.
    epoch_ids = np.searchsorted(events, centres, side='right') - 1
    return windows.astype(np.float32), starts, epoch_ids


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

def kl_mixture_gaussian(mu, logvar, prior_means, prior_logvars, pi):
    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn_like(std)
    log_q = (-0.5 * (LOG_2PI + logvar) - 0.5 * ((z - mu) ** 2) / logvar.exp()).sum(1)

    log_p_components = []
    for k in range(prior_means.size(0)):
        log_pk = (-0.5 * (LOG_2PI + prior_logvars[k])
                  - 0.5 * ((z - prior_means[k]) ** 2) / prior_logvars[k].exp()).sum(1)
        log_p_components.append(torch.log(pi[k]) + log_pk)
    log_p = torch.logsumexp(torch.stack(log_p_components, dim=1), dim=1)
    return (log_q - log_p).mean()


# ──────────────────────── trainer helpers ────────────────────
# def train_mixture_vae(model: nn.Module,
#                       train_loader, val_loader,
#                       prior_means: torch.Tensor,
#                       prior_logvars: torch.Tensor,
#                       pi: torch.Tensor,
#                       epochs: int,
#                       device: torch.device,
#                       *,
#                       beta: float = 1.0,
#                       use_mmd: bool = False,
#                       lam_mmd: float = 10.0,
#                       mmd_sigma: float = 1.0,
#                       verbose: bool = False):
#     """Generic trainer – behaves as VAE (KL) or WAE‑MMD depending on *use_mmd*."""
#     opt = optim.Adam(model.parameters(), lr=5e-4)
#     mse = nn.MSELoss()

#     hist = {'tr_rec': [], 'tr_div': [], 'va_rec': [], 'va_div': []}
#     iters = tqdm(range(epochs)) if verbose else range(epochs)

#     for ep in iters:
#         # ───── training phase ─────
#         model.train()
#         train_Recon_accumulated = train_diver_accumulated = 0.0
#         for (x,) in train_loader:
#             x = x.to(device)
#             recon, mu, lv = model(x)
#             rec_loss = mse(recon, x)

#             if use_mmd:
#                 z_batch = model.reparameterise(mu, lv)
#                 z_prior = sample_mixture_prior(z_batch.size(0),
#                                                prior_means, prior_logvars, pi)
#                 # z_batch = model.reparameterise(mu, lv)
#                 # sample more prior points for stability
#                 # z_prior = sample_mixture_prior(z_batch.size(0)*2, prior_means, prior_logvars, pi)

#                 div_loss = mmd_rbf(z_batch, z_prior, sigma=mmd_sigma)
#                 # div_loss = mmd_multiscale(z_batch, z_prior,
#                 #             sigmas=(0.5,1.0,2.0, 4.0))   # pick scales spanning your mode‐gap
#                 loss = rec_loss + lam_mmd * div_loss
#             else:
#                 div_loss = kl_mixture_gaussian(mu, lv, prior_means, prior_logvars, pi)
#                 loss = rec_loss + beta * div_loss


#             opt.zero_grad(); loss.backward() 

#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             opt.step()
#             bs = x.size(0)
#             train_Recon_accumulated += rec_loss.item() * bs
#             train_diver_accumulated += div_loss.item() * bs

#         # ───── validation phase ─────
#         model.eval()
#         vaR_acc = vaD_acc = 0.0
#         with torch.no_grad():
#             for (x,) in val_loader:
#                 x = x.to(device)
#                 recon, mu, lv = model(x)
#                 rec_loss = mse(recon, x)

#                 if use_mmd:
#                     z_batch = model.reparameterise(mu, lv)
#                     z_prior = sample_mixture_prior(z_batch.size(0),
#                                                    prior_means, prior_logvars, pi)
#                     div_loss = mmd_rbf(z_batch, z_prior, sigma=mmd_sigma)
#                     # z_batch = model.reparameterise(mu, lv)
#                     # # sample more prior points for stability
#                     # z_prior = sample_mixture_prior(z_batch.size(0)*2, prior_means, prior_logvars, pi)

#                     # # div_loss = mmd_rbf(z_batch, z_prior, sigma=mmd_sigma)
#                     # div_loss = mmd_multiscale(z_batch, z_prior,
#                     #             sigmas=(0.5,1.0,2.0))   # pick scales spanning your mode‐gap
#                 else:
#                     div_loss = kl_mixture_gaussian(mu, lv, prior_means, prior_logvars, pi)

#                 bs = x.size(0)
#                 vaR_acc += rec_loss.item() * bs
#                 vaD_acc += div_loss.item() * bs

#         Ntr, Nva = len(train_loader.dataset), len(val_loader.dataset)
#         hist['tr_rec'].append(train_Recon_accumulated / Ntr)
#         hist['tr_div'].append(train_diver_accumulated / Ntr)
#         hist['va_rec'].append(vaR_acc / Nva)
#         hist['va_div'].append(vaD_acc / Nva)

#         if verbose:
#             tag = 'MMD' if use_mmd else 'KL'
#             print(f"E{ep+1:03}/{epochs}  rec {hist['tr_rec'][-1]:.3e}  {tag} {hist['tr_div'][-1]:.3e}")

#     return hist

def train_mixture_vae(model: nn.Module,
                      train_loader, val_loader,
                      prior_means: torch.Tensor,
                      prior_logvars: torch.Tensor,
                      pi: torch.Tensor,
                      epochs: int,
                      device: torch.device,
                      *,
                      beta: float = 1.0,
                      use_mmd: bool = False,
                      lam_mmd: float = 10.0,
                      mmd_sigma: float = 1.0,
                      consistency_weight: float = 0.0,
                      verbose: bool = False):
    """
    Generic trainer – behaves as VAE (KL) or WAE‑MMD depending on *use_mmd*.
    Now also expects the dataloader to yield (x, epoch_ids) and adds a consistency
    loss term. This loss penalizes if windows from the same epoch are assigned
    by the network to different clusters (measured via the soft-assignment based on
    distance from prior means).
    """
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    mse = torch.nn.MSELoss()
    hist = {'tr_rec': [], 'tr_div': [], 'tr_cons': [], 'va_rec': [], 'va_div': []}
    iters = range(epochs)
    if verbose:
        iters = torch.arange(epochs)
    
    for ep in iters:
        model.train()
        train_rec_accum = train_div_accum = train_cons_accum = 0.0
        for batch in train_loader:
            # Unpack batch: x shape (B,C,L) and epoch_ids shape (B,)
            x, epoch_ids = batch
            x = x.to(device)
            epoch_ids = epoch_ids.to(device)
            recon, mu, lv = model(x)
            rec_loss = mse(recon, x)
            
            if use_mmd:
                z_batch = model.reparameterise(mu, lv)
                z_prior = sample_mixture_prior(z_batch.size(0),
                                               prior_means, prior_logvars, pi)
                div_loss = mmd_rbf(z_batch, z_prior, sigma=mmd_sigma)
                loss = rec_loss + lam_mmd * div_loss
            else:
                div_loss = kl_mixture_gaussian(mu, lv, prior_means, prior_logvars, pi)
                loss = rec_loss + beta * div_loss

            # ---- Consistency loss between windows from the same epoch ----
            # compute soft assignments using distances from each prior mean
            # dists shape: (B,2)
            dists = torch.cdist(mu, prior_means)
            assign_logits = -dists  # higher logit for closer prior
            probs = torch.softmax(assign_logits, dim=1)  # (B,2)

            # Group by epoch id
            cons_loss = 0.0
            count = 0
            unique_epochs = epoch_ids.unique()
            for grp in unique_epochs:
                grp_mask = (epoch_ids == grp)
                if grp_mask.sum() < 2:
                    continue  # skip groups too small to compute diversity
                grp_probs = probs[grp_mask]  # shape (n_grp,2)
                grp_mean = grp_probs.mean(dim=0, keepdim=True)
                cons_loss = cons_loss + ((grp_probs - grp_mean)**2).mean()
                count += 1
            if count > 0:
                cons_loss = cons_loss / count
            else:
                cons_loss = 0.0

            loss = loss + consistency_weight * cons_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = x.size(0)
            train_rec_accum += rec_loss.item() * bs
            train_div_accum += div_loss.item() * bs
            train_cons_accum += cons_loss if isinstance(cons_loss, float) else cons_loss.item() * bs

        # Validation phase (consistency loss omitted)
        model.eval()
        va_rec, va_div = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, _ = batch
                x = x.to(device)
                recon, mu, lv = model(x)
                rec_loss = mse(recon, x)
                if use_mmd:
                    z_batch = model.reparameterise(mu, lv)
                    z_prior = sample_mixture_prior(z_batch.size(0),
                                                   prior_means, prior_logvars, pi)
                    div_loss = mmd_rbf(z_batch, z_prior, sigma=mmd_sigma)
                else:
                    div_loss = kl_mixture_gaussian(mu, lv, prior_means, prior_logvars, pi)
                bs = x.size(0)
                va_rec += rec_loss.item() * bs
                va_div += div_loss.item() * bs

        Ntr = len(train_loader.dataset)
        Nva = len(val_loader.dataset)
        hist['tr_rec'].append(train_rec_accum / Ntr)
        hist['tr_div'].append(train_div_accum / Ntr)
        hist['tr_cons'].append(train_cons_accum / Ntr)
        hist['va_rec'].append(va_rec / Nva)
        hist['va_div'].append(va_div / Nva)

        if verbose:
            tag = 'MMD' if use_mmd else 'KL'
            print(f"E{ep+1:03}/{epochs}  rec: {hist['tr_rec'][-1]:.3e}  {tag}: {hist['tr_div'][-1]:.3e}  cons: {hist['tr_cons'][-1]:.3e}")
    return hist