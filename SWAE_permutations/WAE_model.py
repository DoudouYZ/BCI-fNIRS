# ---------------------------------------------------------------------------
#             Wasserstein Auto-Encoder (Sliced-Wasserstein variant)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  PyTorch cube-style SWAE for (n_perm × C × T) datapoints
# ---------------------------------------------------------------------------
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import combinations
from numpy.random import default_rng
import numpy as np

# ---------------------------------------------------------------------------
# Dataset --------------------------------------------------------------------
# ---------------------------------------------------------------------------
class CubeDataset(Dataset):
    """
    Wrap a list of cubes (n_perm, C, T) from `preprocess_for_wae_cube`.
    • Optional z-scoring per channel.
    • Delivers tensor with shape (1, n_perm, C, T) for Conv3d input.
    """
    def __init__(self, cubes, zscore=True):
        self.tensors = []
        for cube in cubes:
            if zscore:
                m = cube.mean(axis=(0, 2), keepdims=True)
                s = cube.std(axis=(0, 2), keepdims=True) + 1e-6
                cube = (cube - m) / s
            self.tensors.append(torch.tensor(cube, dtype=torch.float32).unsqueeze(0))
        self.D, self.C, self.T = self.tensors[0].shape[1:]  # n_perm, C, T

    def __len__(self):          return len(self.tensors)
    def __getitem__(self, idx): return self.tensors[idx]

# ---------------------------------------------------------------------------
# Prior (same as before) ------------------------------------------------------
# ---------------------------------------------------------------------------
class GMMPrior(nn.Module):
    def __init__(self, z_dim, init_mu=2.0, init_logvar=0.0, learnable=True):
        super().__init__()
        mu0, mu1 = torch.full((z_dim,),  init_mu), torch.full((z_dim,), -init_mu)
        self.mu      = nn.Parameter(torch.stack([mu0, mu1]),  requires_grad=learnable)
        self.logvar  = nn.Parameter(torch.full((2, z_dim), init_logvar),
                                    requires_grad=learnable)
        self.register_buffer("weights", torch.tensor([0.5, 0.5]))

    def sample(self, n):
        comp = torch.multinomial(self.weights, n, replacement=True)
        eps  = torch.randn(n, self.mu.size(1), device=self.mu.device)
        return self.mu[comp] + eps * torch.exp(0.5 * self.logvar[comp])

# ---------------------------------------------------------------------------
# Sliced Wasserstein distance (unchanged) ------------------------------------
# ---------------------------------------------------------------------------
def sliced_wasserstein(enc_z, prior_z, n_proj=100, p=2):
    B, z_dim = enc_z.shape
    dirs = torch.randn(n_proj, z_dim, device=enc_z.device)
    dirs = dirs / torch.norm(dirs, dim=1, keepdim=True)
    s_enc   = torch.sort(enc_z  @ dirs.T, dim=0)[0]
    s_prior = torch.sort(prior_z @ dirs.T, dim=0)[0]
    return torch.mean(torch.abs(s_enc - s_prior) ** p)

# ---------------------------------------------------------------------------
# Encoder / Decoder (3-D) -----------------------------------------------------
# ---------------------------------------------------------------------------
class CubeEncoder(nn.Module):
    """
    Input shape  (B, 1, n_perm, C, T)
    Uses grouped first conv so HbO/HbR channels stay separate initially.
    """
    def __init__(self, n_perm, C, z_dim=8):
        super().__init__()
        # in_channels = 1
        self.block1 = nn.Conv3d(1, 32,
                                kernel_size=(1, 1, 15),
                                stride=(1, 1, 3),
                                padding=(0, 0, 7))
        # sweep across all epochs and a small channel neighbourhood
        self.block2 = nn.Conv3d(32, 64,
                                kernel_size=(n_perm, 3, 11),
                                stride=(1, 1, 3),
                                padding=(0, 1, 5))
        # collapse channel dim
        self.block3 = nn.Conv3d(64, 128,
                                kernel_size=(1, C, 1))
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.mu     = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

    def forward(self, x):
        x = F.gelu(self.block1(x))
        x = F.gelu(self.block2(x))
        x = F.gelu(self.block3(x))
        x = self.pool(x).flatten(1)
        return self.mu(x), self.logvar(x)


class CubeDecoder(nn.Module):
    """
    Mirrors encoder via FC → reshape to cube.
    Output shape matches input: (B, 1, n_perm, C, T)
    """
    def __init__(self, n_perm, C, T, z_dim=8):
        super().__init__()
        self.n_perm, self.C, self.T = n_perm, C, T
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_perm * C * T)
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 1, self.n_perm, self.C, self.T)
        return x

# ---------------------------------------------------------------------------
# Trainer --------------------------------------------------------------------
# ---------------------------------------------------------------------------
def train_swae_cube(cubes,
                    z_dim=8,
                    batch_size=8,
                    lr=1e-3,
                    epochs=200,
                    lamda=25.0,
                    n_proj=100,
                    device=None):
    """
    cubes : list of np.ndarray, each (n_perm, C, T)
    Returns : encoder, decoder, prior
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds  = CubeDataset(cubes)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    enc = CubeEncoder(ds.D, ds.C, z_dim).to(device)
    dec = CubeDecoder(ds.D, ds.C, ds.T, z_dim).to(device)
    pri = GMMPrior(z_dim).to(device)

    opt   = torch.optim.Adam(list(enc.parameters()) +
                             list(dec.parameters()) +
                             list(pri.parameters()), lr=lr)
    mse   = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        recon_total, swd_total = 0.0, 0.0
        for x in dl:
            x = x.to(device)                     # (B,1,D,C,T)
            mu, logv = enc(x)
            z        = mu + torch.exp(0.5*logv) * torch.randn_like(mu)
            x_hat    = dec(z)

            loss_recon = mse(x_hat, x)
            loss_swd   = sliced_wasserstein(z, pri.sample(x.size(0)),
                                            n_proj=n_proj)
            loss = loss_recon + lamda * loss_swd

            opt.zero_grad()
            loss.backward()
            opt.step()

            recon_total += loss_recon.item()
            swd_total   += loss_swd.item()

        print(f"Epoch {epoch:3d} | recon {recon_total/len(dl):.4f} | "
              f"swd {swd_total/len(dl):.4f}")

    return enc, dec, pri
