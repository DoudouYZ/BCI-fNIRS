import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is a starter file for experimenting with a VAE using a mixture-of-Gaussians prior.
# We'll build it up in steps:
# 1. Define a simple encoder that outputs mean and log-variance.
# 2. Define a decoder that reconstructs inputs from latent z.
# 3. Write the KL divergence to a mixture-of-2-Gaussians prior as a function.
# 4. Hook up the loss and a simple training loop on dummy data.

class Encoder(nn.Module):
    def __init__(self, input_channels: int, input_length: int, latent_dim: int):
        """
        Args:
            input_channels (int): Number of input channels (e.g. 40).
            input_length (int): Number of time steps (e.g. 79).
            latent_dim (int): Dimension of the latent representation.
        """
        last_channel_n = 16
        super(Encoder, self).__init__()
        # First convolution: keep time dimension same
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16,
                               kernel_size=11, stride=1, padding=5)
        # Second convolution: downsample by factor 2
        self.avg_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=7, stride=1, padding=3)
        self.avg_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Third convolution: further downsample by factor 2
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=last_channel_n,
                               kernel_size=3, stride=1, padding=1)
        
        # Compute the output length after conv layers using the formula:
        # L_out = floor((L + 2*padding - kernel_size)/stride + 1)
        def conv_out_length(L, kernel_size=3, stride=1, padding=1):
            return (L + 2 * padding - kernel_size) // stride + 1
        
        L = conv_out_length(input_length, kernel_size=11, stride=1, padding=5)

        L = conv_out_length(L, kernel_size=2, stride=2, padding=0)

        L = conv_out_length(L, kernel_size=7, stride=1, padding=3)

        L = conv_out_length(L, kernel_size=2, stride=2, padding=0)
        
        L = conv_out_length(L, kernel_size=3, stride=1, padding=1)
        self.conv_output_length = L  # save for the decoder
        
        self.flattened_size = last_channel_n * L
        self.encoder_fc1 = nn.Linear(self.flattened_size, self.flattened_size//2)
        self.encoder_dropout = nn.Dropout(0.5)
        self.encoder_fc_mu = nn.Linear(self.flattened_size//2, latent_dim)
        self.encoder_fc_log_var = nn.Linear(self.flattened_size//2, latent_dim)

    def forward(self, x):

        ############################
        # Should return (mu, logvar) each of size [batch, latent_dim]
        ############################

        # x: (batch, input_channels, input_length)
        x = F.relu(self.conv1(x))     # -> (batch, 32, L)
        x = self.avg_pool1(x)
        x = F.relu(self.conv2(x))     # -> (batch, 32, L2)
        x = self.avg_pool2(x)
        x = F.relu(self.conv3(x))     # -> (batch, 16, L3)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)    # flatten
        x = self.encoder_dropout(x)
        x = F.sigmoid(self.encoder_fc1(x))
        x = self.encoder_dropout(x)
        mu = self.encoder_fc_mu(x)           # -> (batch, latent_dim)
        log_var = self.encoder_fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, input_channels: int, input_length: int, latent_dim: int):
        super().__init__()
        # TODO: add layers
        last_channel_n = 16
        # ---------- encoder ----------
        self.encoder = Encoder(input_channels, input_length, latent_dim)

        # ---------- decoder ----------
        L_enc = self.encoder.conv_output_length          # == input_length // 4
        self.L_enc = L_enc                                # save for debugging
        self.flat_size = last_channel_n * L_enc

        # 1) Latent → dense → (batch, 8, L_enc)
        self.dec_fc = nn.Linear(latent_dim, self.flat_size)

        # 2) Upsample + conv mirrors
        # -- first upsample (reverse pool2) --
        self.up1   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_t1 = nn.Conv1d(last_channel_n, 32, kernel_size=3, stride=1, padding=1)

        # -- second upsample (reverse pool1) --
        self.up2   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_t2 = nn.Conv1d(32, 16, kernel_size=7, stride=1, padding=3)

        # -- final convolution (reverse conv1) --
        self.conv_t3 = nn.Conv1d(16, input_channels, kernel_size=11,
                                 stride=1, padding=5)

    def forward(self, z):
        # Should return reconstructions of shape [batch, output_dim]
        """
        x shape  : (batch, channels, time)
        returns  : reconstruction same shape
        """
        # latent → dense → reshape
        x_hat = self.dec_fc(z)
        x_hat = x_hat.view(x_hat.size(0), 16, self.L_enc)

        # upsample-conv-relu  (mirror sequence)
        x_hat = F.relu(self.conv_t1(self.up1(x_hat)))
        x_hat = F.relu(self.conv_t2(self.up2(x_hat)))
        x_hat = self.conv_t3(x_hat)             # last layer linear activation
        return x_hat

# Mixture-of-Gaussians KL term
# Inputs:
#   mu:        [batch, latent_dim]
#   logvar:    [batch, latent_dim]
#   pis:       [2] mixture weights (sum to 1)\#   mus:       [2, latent_dim] component means
#   logvars:   [2, latent_dim] component log-variances
# Returns:
#   kl:        [batch] the KL divergence for each example
class VAE(nn.Module):
    def __init__(self, input_channels: int, input_length: int, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_channels, input_length, latent_dim)
        self.decoder = Decoder(input_channels, input_length, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        # sample once for reconstruction
        std  = torch.exp(0.5 * logvar)
        z    = mu + std * torch.randn_like(std)
        xhat = self.decoder(z)
        return xhat, mu, logvar

def kl_mixture_gaussian(mu, logvar,
                        prior_means,
                        prior_logvars=None,
                        pi=None,
                        n_samples=1):
    """
    Monte-Carlo KL( q(z|x)  ||  p_mix(z) )     — draws the z-sample(s) internally.

    Args
    ----
    mu, logvar   : [batch, D]  encoder outputs
    prior_means  : [K,   D]    mixture component centres
    prior_logvars: [K,   D] or broadcastable  — component log-variances (default 0)
    pi           : [K]         mixture weights (default uniform)
    n_samples    : int         how many z-draws per example (1 = standard ELBO)

    Returns
    -------
    kl_mean : scalar  — mean KL over batch (and samples)
    """
    batch, D = mu.shape
    K, _     = prior_means.shape

    # ----- 1. sample z from q(z|x)  -----
    std   = torch.exp(0.5 * logvar)              # σ
    eps   = torch.randn((n_samples, batch, D),
                        device=mu.device,
                        dtype=mu.dtype)          # ε ~ N(0,1)
    z     = mu.unsqueeze(0) + std.unsqueeze(0) * eps   # [S, B, D]

    # ----- 2. log q(z|x) -----
    #   -0.5 * [ log(2πσ²) + (z-μ)²/σ² ]  summed over D
    log_q = -0.5 * (
        (logvar + math.log(2*math.pi)).sum(dim=1, keepdim=True)  # [B,1]
        + ((z - mu.unsqueeze(0))**2 / torch.exp(logvar).unsqueeze(0)).sum(dim=2)
    )  # [S, B]

    # ----- 3. log p_mix(z) -----
    if prior_logvars is None:
        prior_logvars = torch.zeros_like(prior_means)            # default unit var
    if pi is None:
        pi = torch.full((K,), 1.0/K, device=mu.device)

    z_e   = z.unsqueeze(2)                    # [S,B,1,D]
    mu_k  = prior_means.unsqueeze(0).unsqueeze(0)   # [1,1,K,D]
    var_k = prior_logvars.exp().unsqueeze(0).unsqueeze(0)        # [1,1,K,D]

    # log N_k(z)
    log_Nk = -0.5 * (
        (prior_logvars + math.log(2*math.pi)).sum(dim=1)    # [K]
        + ((z_e - mu_k)**2 / var_k).sum(dim=3)              # [S,B,K]
    )  # [S,B,K]

    log_pi = torch.log(pi).view(1,1,K)        # [1,1,K]
    log_components = log_pi + log_Nk          # [S,B,K]
    log_p = torch.logsumexp(log_components, dim=2)    # [S,B]

    # ----- 4. KL per sample then average -----
    kl = (log_q - log_p).mean()               # scalar
    return kl
def plot_latent_space(model, data_loader, device):
    """
    Runs the model.encoder over all batches in data_loader, collects
    the latent means (mu) and makes a scatter of the first two dims.
    
    Args:
        model      : your trained VAE instance
        data_loader: torch.utils.data.DataLoader yielding input batches
        device     : 'cpu' or 'cuda'
    """
    model.eval()
    all_mu = []
    with torch.no_grad():
        for batch in data_loader:
            # batch could be (x,) or x depending on your loader
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            mu, _ = model.encoder(x)
            all_mu.append(mu.cpu())
    all_mu = torch.cat(all_mu, dim=0).numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(all_mu[:, 0], all_mu[:, 1],
                s=10, alpha=0.7, edgecolors='none')
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.title('Encoder Means in Latent Space')
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    plt.tight_layout()
    plt.show()

def make_example_data(n_train, n_val, n_channels, n_times):

    # time‐axis from 0→1 second
    t = torch.linspace(0, 1, n_times).unsqueeze(0).unsqueeze(0)  # shape [1,1,T]

    # pick one frequency and random phase per channel
    freqs  = torch.linspace(1.0, 3.0, n_channels).view(1, n_channels, 1)  # [1,C,1]
    phases = torch.rand(1, n_channels, 1) * 2 * math.pi                # [1,C,1]

    # ─────────────── generate train ───────────────
    # binary labels 0/1
    train_labels = torch.randint(0, 2, (n_train,))

    # amplitudes: class 0 → 1.0, class 1 → 2.5
    train_amps = torch.where(train_labels.unsqueeze(1)==0,
                            torch.tensor(1.0),
                            torch.tensor(2.5))             # [N,1]

    # build the sine waves + small noise
    train_data = train_amps.unsqueeze(2) * torch.sin(2*math.pi*freqs*t + phases) \
                + 0.05*torch.randn(n_train, n_channels, n_times)

    # ─────────────── generate val ───────────────
    val_labels = torch.randint(0, 2, (n_val,))
    val_amps   = torch.where(val_labels.unsqueeze(1)==0,
                            torch.tensor(1.0),
                            torch.tensor(2.5))
    val_data = val_amps.unsqueeze(2) * torch.sin(2*math.pi*freqs*t + phases) \
            + 0.05*torch.randn(n_val, n_channels, n_times)

    # ─────────────── wrap in DataLoaders ───────────────
    batch_size = 64
    train_loader = DataLoader(
        TensorDataset(train_data, train_labels),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_data, val_labels),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader

    # now train_loader yields (x, y) where x.shape == (B,40,32)
    # and y in {0,1}, with class‐1 having ~2.5× larger sine amplitude.

# ─────────────── parameters ───────────────
n_train, n_val = 1024, 256
n_channels     = 40
n_times        = 32

train_loader, val_loader = make_example_data(n_train, n_val, n_channels, n_times)
# ────────────────────────────────────────────────────────────────

# Then instantiate your VAE with matching dimensions:
vae = VAE(input_channels=n_channels,
          input_length=n_times,
          latent_dim=2).to(device)

opt   = torch.optim.Adam(vae.parameters(), lr=1e-3)
mse   = nn.MSELoss(reduction="mean")


beta  = 0.5        # weight on the KL term
prior_means    = torch.tensor([[ 2, 2],
                               [-2, -2]], device=device)
prior_logvars  = torch.zeros_like(prior_means)          # unit variance
pi_mix         = torch.tensor([0.5, 0.5], device=device)


EPOCHS = 30
for epoch in range(1, EPOCHS+1):
    vae.train()
    rec_sum = kl_sum = 0.0

    for x, _ in train_loader:
        x = x.to(device)

        # forward pass
        recon, mu, logvar = vae(x)
        rec_loss = mse(recon, x)

        kl_loss = kl_mixture_gaussian(
            mu, logvar,
            prior_means, prior_logvars, pi_mix
        )

        loss = rec_loss + beta * kl_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        rec_sum += rec_loss.item() * x.size(0)
        kl_sum  += kl_loss.item()  * x.size(0)

    n_train   = len(train_loader.dataset)
    avg_rec   = rec_sum / n_train
    avg_kl    = kl_sum  / n_train
    avg_total = avg_rec + beta * avg_kl

    # ---- quick validation MSE ----
    vae.eval()
    with torch.no_grad():
        mse_val = 0.0
        for x, _ in val_loader:
            x = x.to(device)
            recon, _, _ = vae(x)
            mse_val += mse(recon, x).item() * x.size(0)
        mse_val /= len(val_loader.dataset)

    print(f"E{epoch:02d}: train-MSE {avg_rec:.4e}  train-KL {avg_kl:.4e}  "
          f"val-MSE {mse_val:.4e}")
    
plot_latent_space(vae, train_loader, device)
