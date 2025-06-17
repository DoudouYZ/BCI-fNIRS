"""
1-D convolutional SWAE for (C,T) epochs.
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# ────────────────────────────────────────────────────────────────────────────
#  DATASET
# ────────────────────────────────────────────────────────────────────────────
class EpochDataset(Dataset):
    def __init__(self, arrays):
        self.arrays = [torch.tensor(a, dtype=torch.float32)
                       for a in arrays]            # (C,T)
        self.C, self.T = self.arrays[0].shape

    def __len__(self):          return len(self.arrays)
    def __getitem__(self, idx): return self.arrays[idx]          # (C,T)

# ────────────────────────────────────────────────────────────────────────────
#  PRIOR
# ────────────────────────────────────────────────────────────────────────────
class GMMPrior(nn.Module):
    def __init__(self, z_dim=4, init=0.5, learnable=False):
        super().__init__()
        mu0 = torch.full((z_dim,),  init)
        mu1 = torch.full((z_dim,), -init)
        if learnable:
            self.mu     = nn.Parameter(torch.stack([mu0, mu1]))      # (2, z_dim)
            self.logvar = nn.Parameter(torch.zeros(2, z_dim))        # (2, z_dim)
        else:
            self.register_buffer("mu", torch.stack([mu0, mu1]))      
            self.register_buffer("logvar", torch.zeros(2, z_dim))    
        self.register_buffer("w", torch.tensor([0.5, 0.5]))      # fixed mix-weights

    def sample(self, n):
        comp = torch.multinomial(self.w, n, replacement=True)    # (n,)
        eps  = torch.randn(n, self.mu.size(1), device=self.mu.device)
        return self.mu[comp] + eps * torch.exp(0.5 * self.logvar[comp])


# ────────────────────────────────────────────────────────────────────────────
#  SLICED WASSERSTEIN
# ────────────────────────────────────────────────────────────────────────────
def sliced_wasserstein(z, z_p, n_proj=100):
    B, D = z.shape
    dirs = torch.randn(n_proj, D, device=z.device)
    dirs = dirs/dirs.norm(dim=1, keepdim=True)
    zs  = torch.sort(z  @ dirs.T, dim=0)[0]
    zps = torch.sort(z_p@ dirs.T, dim=0)[0]
    return (zs - zps).abs().pow(2).mean()

# ────────────────────────────────────────────────────────────────────────────
#  ENCODER / DECODER
# ────────────────────────────────────────────────────────────────────────────
class Encoder1D(nn.Module):
    def __init__(self, C, z_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 64, kernel_size=15, stride=3, padding=7, groups=2),
            nn.GELU(),
            nn.Conv1d(64,128,kernel_size=11,stride=3,padding=5),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mu     = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)

    def forward(self,x):                     # x (B,C,T)
        h = self.net(x).squeeze(-1)          # (B,128)
        return self.mu(h), self.logvar(h)

class Decoder1D(nn.Module):
    def __init__(self, C, T, z_dim=4):
        super().__init__()
        self.C, self.T = C, T
        self.fc = nn.Sequential(
            nn.Linear(z_dim,128),
            nn.GELU(),
            nn.Linear(128,C*T),
        )

    def forward(self,z):
        x = self.fc(z).view(-1,self.C,self.T)   # (B,C,T)
        return x
