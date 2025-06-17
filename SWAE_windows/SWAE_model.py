"""
2-D Sliced-Wasserstein Auto-Encoder
Input tensor shape :  (B, 1, C, T)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# dataset
# ────────────────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """
    Wrap a list of (C,T) windows.
    • Per-channel z-score **ignores the zero-padded region**.
    • Returns tensor (1, C, T).
    """
    def __init__(self, matrices, zscore=True):
        self.tensors = []
        for m in matrices:                      # m : (C,T) ndarray
            if zscore:
                mask = m != 0
                # mean & std over non-zero time points for each channel
                mu  = np.ma.array(m, mask=~mask).mean(axis=1).filled(0).reshape(-1,1)
                sd  = np.ma.array(m, mask=~mask).std(axis=1).filled(0).reshape(-1,1)
                # Replace any zero std by a small constant to avoid division by zero
                sd = np.where(sd==0, 1e-6, sd)
                m   = (m - mu) / sd
            self.tensors.append(torch.tensor(m, dtype=torch.float32).unsqueeze(0))
        # cache C, T once
        self.C, self.T = self.tensors[0].shape[-2:]

    def __len__(self):          return len(self.tensors)
    def __getitem__(self, idx): return self.tensors[idx]



# ────────────────────────────────────────────────────────────────────────────
# prior – two-component GMM  (unchanged)
# ────────────────────────────────────────────────────────────────────────────
class GMMPrior(nn.Module):
    def __init__(self, z_dim, mu=0.7, logvar=0.0, learnable=True):
        super().__init__()
        mu_tensor = torch.tensor([[mu], [-mu]])
        self.mu = nn.Parameter(mu_tensor.repeat(1, z_dim), requires_grad=learnable)
        logvar_tensor = torch.tensor([[logvar], [logvar]])
        self.logvar = nn.Parameter(logvar_tensor.repeat(1, z_dim), requires_grad=learnable)
        self.register_buffer("w", torch.tensor([0.5, 0.5]))

    def sample(self, n):
        comp = torch.multinomial(self.w, n, replacement=True)
        eps  = torch.randn(n, self.mu.size(1), device=self.mu.device)
        return self.mu[comp] + eps * torch.exp(0.5 * self.logvar[comp])
    
    def clamp_parameters(self, mu_min, mu_max, logvar_min, logvar_max):
        self.mu.data.clamp_(mu_min, mu_max)
        self.logvar.data.clamp_(logvar_min, logvar_max)

# ────────────────────────────────────────────────────────────────────────────
# distance
# ────────────────────────────────────────────────────────────────────────────
def sliced_wasserstein(z, z_prior, n_proj=100):
    dirs = torch.randn(n_proj, z.size(1), device=z.device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    proj1 = torch.sort(z       @ dirs.T, dim=0)[0]
    proj2 = torch.sort(z_prior @ dirs.T, dim=0)[0]
    return (proj1 - proj2).abs().pow(2).mean()


# ────────────────────────────────────────────────────────────────────────────
# encoder / decoder  (Conv2d)
# ────────────────────────────────────────────────────────────────────────────
class Encoder2D(nn.Module):
    """
    Input  (B, 1, C, T)
    Output two tensors (μ, logσ²) each of shape (B, z_dim)
    """
    def __init__(self, C, z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 15), stride=(1, 3), padding=(0, 7)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=(3, 11), stride=(1, 3), padding=(1, 5)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=(C, 1)),  # collapse channel dim
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 128, 1, 1)
        self.mu     = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

    def forward(self, x):
        h = self.conv(x)             # (B,128,·,·)
        h = self.pool(h).flatten(1)   # (B,128)
        mu = self.mu(h)
        logv = self.logvar(h)
        # Clamp to avoid NaNs (do not clamp the prior parameters)
        mu = torch.clamp(mu, min=-100.0, max=100.0)
        logv = torch.clamp(logv, min=-10.0, max=10.0)
        return mu, logv

class Decoder2D(nn.Module):
    def __init__(self, C, T, z_dim):
        super().__init__()
        self.C, self.T = C, T
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, C * T)
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 1, self.C, self.T)
        return x


# ────────────────────────────────────────────────────────────────────────────
# trainer helper
# ────────────────────────────────────────────────────────────────────────────
def build_dataloader(matrices, batch, shuffle=True, drop_last=False):
    return DataLoader(WindowDataset(matrices),
                      batch_size=batch,
                      shuffle=shuffle,
                      drop_last=drop_last)
