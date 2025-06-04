import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm

# -------------------------------
# Helper: Cropping1D Module
# -------------------------------
class Cropping1D(nn.Module):
    def __init__(self, crop_right):
        """
        Crops the last 'crop_right' time steps from the input.
        """
        super(Cropping1D, self).__init__()
        self.crop_right = crop_right

    def forward(self, x):
        # x shape: (batch, channels, time)
        if self.crop_right > 0:
            return x[:, :, :-self.crop_right]
        return x

# -------------------------------
# Encoder Module
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        """
        Args:
            input_channels (int): Number of input channels (e.g. 40).
            input_length (int): Number of time steps (e.g. 79).
            latent_dim (int): Dimension of the latent representation.
        """
        super(Encoder, self).__init__()
        # First convolution: keep time dimension same
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16,
                               kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16,
                               kernel_size=9, stride=1, padding=4)
        # Second convolution: downsample by factor 2
        self.avg_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32,
                               kernel_size=7, stride=1, padding=3)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=7, stride=1, padding=3)
        self.avg_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Third convolution: further downsample by factor 2
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=32,
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
        
        self.flattened_size = 32 * L
        self.encoder_fc1 = nn.Linear(self.flattened_size, self.flattened_size)
        self.encoder_dropout = nn.Dropout(0.5)
        self.encoder_fc2 = nn.Linear(self.flattened_size, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_channels, input_length)
        x = F.relu(self.conv1(x))     # -> (batch, 32, L)
        x = F.relu(self.conv2(x))     # -> (batch, 32, L)
        x = self.avg_pool1(x)
        x = F.relu(self.conv3(x))     # -> (batch, 32, L2)
        x = F.relu(self.conv4(x))     # -> (batch, 32, L)
        x = self.avg_pool2(x)
        x = F.relu(self.conv5(x))     # -> (batch, 16, L3)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)    # flatten
        x = self.encoder_dropout(x)
        x = F.relu(self.encoder_fc1(x))
        x = self.encoder_dropout(x)
        latent = self.encoder_fc2(x)           # -> (batch, latent_dim)
        return latent

# -------------------------------
# Classification Autoencoder
# -------------------------------
class ClassificationAutoencoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim, num_classes):
        """
        Uses the same encoder as above but attaches a classification head.
        """
        super(ClassificationAutoencoder, self).__init__()
        self.encoder = Encoder(input_channels, input_length, latent_dim)
        self.fc1 = nn.Linear(latent_dim, 4)
        self.fc2 = nn.Linear(4, num_classes)
    
    def forward(self, x):
        z = self.encoder(x)
        z = F.relu(self.fc1(z))
        logits = self.fc2(z)  # raw logits; use CrossEntropyLoss which applies softmax internally
        return logits
    
    def get_latent(self, x):
        return self.encoder(x)


# -------------------------------
# Reconstruction Auto-Encoder
# -------------------------------
class ReconstructionAutoencoder(nn.Module):
    """
    Uses the same Encoder you already have and a symmetric decoder that
    up-samples + convolves to rebuild the original (n_channels, n_times) input.
    """
    def __init__(self, input_channels: int, input_length: int, latent_dim: int):
        super().__init__()

        # ------------------ encoder ------------------
        self.encoder = Encoder(input_channels, input_length, latent_dim)
        L_enc      = self.encoder.conv_output_length   # time-length after encoder
        flat_size  = 16 * L_enc                        # 16 channels * L_enc

        # ------------------ decoder ------------------
        # We'll build it as a single nn.Sequential so you can just call self.decoder(z)
        layers = []
        # 1) dense → (batch, 16 * L_enc), then reshape → (batch,16,L_enc)
        layers += [
            nn.Linear(latent_dim, flat_size),
            # PyTorch >=1.8 has Unflatten; if you don't have it, replace with a lambda in forward
            nn.Unflatten(1, (16, L_enc))
        ]
        # 2) upsample + conv + ReLU (reverse of conv3)
        layers += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        # 3) upsample + conv + ReLU (reverse of conv2 + pool2)
        layers += [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        ]
        # 4) final conv (reverse of conv1)
        layers += [
            nn.Conv1d(16, input_channels, kernel_size=11, padding=5),
        ]
        # 5) optional crop if we overshot in time
        out_len = L_enc * 4
        if out_len > input_length:
            excess = out_len - input_length
            layers.append(Cropping1D(crop_right=excess))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # 1) encode to latent
        z = self.encoder(x)              # -> (batch, latent_dim)
        # 2) decode from latent
        x_hat = self.decoder(z)          # -> (batch, channels, time)
        return x_hat

    def get_latent(self, x):
        return self.encoder(x)


# ─────────────────────────  Mixture-prior VAE  ──────────────────────────
class MixtureVAE(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        super().__init__()
        # 1a) Reuse your conv‐stack from Encoder, but stop before its final fc to latent
        self.backbone = Encoder(input_channels, input_length, latent_dim)
        # The backbone currently ends in a single Linear→latent_dim.
        # We want instead two heads (mu & logvar). So swap that single head out:
        hidden_size = self.backbone.encoder_fc2.in_features  # size before latent
        # Remove the old single‐head:
        self.backbone.encoder_fc2 = nn.Identity()
        # New heads:
        self.mu_head     = nn.Linear(hidden_size, latent_dim)
        self.logvar_head = nn.Linear(hidden_size, latent_dim)

        # 1b) Reuse your full ReconstructionAutoencoder’s decoder
        #     we only need its decoder submodule.
        self.decoder = ReconstructionAutoencoder(
            input_channels, input_length, latent_dim
        ).decoder

    def encode(self, x):
        """
        Returns (mu, logvar) both shape [batch, latent_dim]
        """
        # Pass through all convs + fc1 + dropout (the backbone minus final head)
        h = self.backbone(x)              # shape [batch, hidden_size]
        mu     = self.mu_head(h)          # [batch, latent_dim]
        logvar = self.logvar_head(h)      # [batch, latent_dim]
        return mu, logvar

    def forward(self, x):
        # 1) encode to distribution parameters
        mu, logvar = self.encode(x)
        # 2) reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z   = mu + std * eps
        # 3) decode
        recon = self.decoder(z)
        return recon, mu, logvar


# ───────────────── kl-divergence to a 2-component Gaussian mixture ──────
LOG_2PI = math.log(2 * math.pi)

def kl_mixture_gaussian(mu, logvar,
                        prior_means,          # [K, D]
                        prior_logvars,        # [K, D]
                        pi,                   # [K]
                        n_samples=1):
    """
    Single-sample Monte-Carlo KL(q || p_mix).  Shapes:
        mu, logvar : [B, D]
        prior_*    : see above (K=2 here)
    """
    B, D = mu.shape
    K    = prior_means.shape[0]

    # ---- sample z ~ q(z|x) ---------------------------
    std = torch.exp(0.5 * logvar)
    eps = torch.randn((n_samples, B, D), device=mu.device)
    z   = mu.unsqueeze(0) + std.unsqueeze(0) * eps        # [S,B,D]

    # ---- log q(z|x)  (diagonal Gauss) -----------------
    log_q = -0.5 * (
        (logvar + LOG_2PI).sum(dim=1, keepdim=True) +
        ((z - mu.unsqueeze(0)) ** 2 / logvar.exp().unsqueeze(0)).sum(dim=2)
    )                                                      # [S,B]

    # ---- log p_mix(z) ---------------------------------
    z_e   = z.unsqueeze(2)                                 # [S,B,1,D]
    mu_k  = prior_means.unsqueeze(0).unsqueeze(0)          # [1,1,K,D]
    var_k = prior_logvars.exp().unsqueeze(0).unsqueeze(0)  # [1,1,K,D]

    log_Nk = -0.5 * (
        (prior_logvars + LOG_2PI).sum(dim=1) +
        ((z_e - mu_k) ** 2 / var_k).sum(dim=3)
    )                                                      # [S,B,K]

    log_pi = torch.log(pi).view(1, 1, K)
    log_p  = torch.logsumexp(log_pi + log_Nk, dim=2)       # [S,B]

    # ---- KL  -----------------------------------------
    return (log_q - log_p).mean()        # scalar
# -------------------------------
# Training Functions
# -------------------------------

def train_classification_autoencoder(model, train_loader, val_loader, epochs, device, verbose=False, transform_data=False, noise_std=0.005, subtract_max=0.001, scale_max=0.01, shift_max=1):
    """
    Training loop for the classification autoencoder.
    """
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    epoch_iter = tqdm(range(epochs)) if verbose else range(epochs)
    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            if transform_data:
                inputs = augment_fnirs_data(inputs, noise_std=noise_std, subtract_max=subtract_max, scale_max=scale_max, shift_max=shift_max)
            inputs = inputs.to(device)
            targets = targets.to(device)  # targets should be LongTensor (class indices)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if verbose:
            print(f"Classification AE Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    return history


def train_reconstruction_autoencoder(model, train_loader, val_loader,
                                     epochs, device, lr=1e-3, verbose=False):
    model.to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    mse  = nn.MSELoss()

    history = {'tr_loss': [], 'va_loss': []}
    itr = tqdm(range(epochs)) if verbose else range(epochs)

    for ep in itr:
        # -------- train --------
        model.train()
        running = 0.0
        for (x,) in train_loader:              # loader yields (inputs,)
            x = x.to(device)
            opt.zero_grad()
            out = model(x)
            loss = mse(out, x)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        tr_loss = running / len(train_loader.dataset)

        # -------- val ----------
        model.eval()
        running = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                running += mse(model(x), x).item() * x.size(0)
        va_loss = running / len(val_loader.dataset)

        history['tr_loss'].append(tr_loss)
        history['va_loss'].append(va_loss)
        if verbose:
            print(f'E{ep+1:02d}/{epochs}  train {tr_loss:.4e}  val {va_loss:.4e}')

    return history

def train_mixture_vae(model, train_loader, val_loader,
                      prior_means, prior_logvars, pi_mix,
                      epochs, device, beta=1.0, verbose=False):

    opt = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    hist = {'tr_rec': [], 
            'tr_kl': [], 
            'va_rec': [], 
            'va_kl': []}

    rng = tqdm(range(epochs)) if verbose else range(epochs)
    for i, ep in enumerate(rng):
        # ---- train ----
        model.train()
        rec_sum = kl_sum = 0.
        for (x,) in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            rec = mse(recon, x)
            kl  = kl_mixture_gaussian(mu, logvar,
                                    prior_means, prior_logvars, pi_mix)
            loss = rec + beta * kl
        
            opt.zero_grad(); loss.backward(); opt.step()
            rec_sum += rec.item() * x.size(0)
            kl_sum  += kl.item()  * x.size(0)

        # ---- val  ----
        model.eval()
        rec_val = kl_val = 0.
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                rec_val += mse(recon, x).item() * x.size(0)
                kl_val  += kl_mixture_gaussian(mu, logvar,
                                               prior_means, prior_logvars,
                                               pi_mix).item() * x.size(0)

        Ntr, Nva = len(train_loader.dataset), len(val_loader.dataset)
        hist['tr_rec'].append(rec_sum/Ntr); hist['tr_kl'].append(kl_sum/Ntr)
        hist['va_rec'].append(rec_val/Nva);  hist['va_kl'].append(kl_val/Nva)

        if verbose:
            print(f"E{ep+1:04d}/{epochs}  rec {hist['tr_rec'][-1]:.3e}  "
                  f"kl {hist['tr_kl'][-1]:.3e}")
    return hist


def create_sliding_windows(data, labels, window_length):
    """
    Splits each epoch in each array in 'data' into overlapping windows using a sliding window approach,
    and replicates the corresponding label for each new window.
    
    Parameters:
        data (list of np.ndarray): Each element has shape (n_epochs, n_channels, n_times).
                                   Note that n_epochs and n_times may differ between arrays.
        labels (list or np.ndarray): A list of labels corresponding to each array in data.
                                     (Each array's epochs are all assigned the same label.)
        window_length (int): Number of observations (time points) per window.
        
    Returns:
        windows (np.ndarray): Array with shape 
            (total_new_samples, n_channels, window_length),
            where total_new_samples is the sum over arrays of (n_epochs * (n_times - window_length + 1)).
        new_labels (np.ndarray): 1D array of labels of length total_new_samples.
            For each epoch, the first (n_windows // 5) and the last (n_windows // 😎 windows get label 0,
            and the middle windows get the original label.
    """
    windows_all = []
    new_labels = []
    # Iterate over each array in data and its corresponding label.
    for X, label in zip(data, labels):
        # X has shape (n_epochs, n_channels, n_times)
        # Create sliding windows along the time axis.
        # This returns an array of shape (n_epochs, n_channels, n_windows, window_length),
        # where n_windows = n_times - window_length + 1.
        windows = np.lib.stride_tricks.sliding_window_view(X, window_length, axis=2)
        
        # Transpose to get shape: (n_epochs, n_windows, n_channels, window_length)
        windows = windows.transpose(0, 2, 1, 3)
        
        # Get dimensions for the current array.
        n_epochs, n_windows, n_channels, _ = windows.shape
        
        # Reshape windows so each sliding window becomes an independent sample:
        # New shape: (n_epochs * n_windows, n_channels, window_length)
        windows_reshaped = windows.reshape(n_epochs * n_windows, n_channels, window_length)
        windows_all.append(windows_reshaped)
        
        # Compute cutoff indices for the current array.
        # These determine how many windows at the beginning and end get assigned a label of 0.
        start_cut = n_windows // 5
        end_cut = 0
        
        # For each epoch in the current array, create a label vector for its windows.
        for i in label:
            new_labels.extend([0] * start_cut + [i] * (n_windows - start_cut - end_cut) + [0] * (end_cut))
        
    # Concatenate windows from all arrays into a single array.
    windows_all = np.concatenate(windows_all, axis=0)
    new_labels = np.array(new_labels)  # This is a 1D array.
    
    return windows_all.astype(np.float32), new_labels

def create_sliding_windows_no_classes(data, window_length):
    """
    Split a continuous multichannel time series into overlapping windows.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
        The continuous signal for each channel.
    window_length : int
        Number of timepoints per sliding window.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, n_channels, window_length)
        Overlapping windows, where
          n_windows = n_times - window_length + 1.
    """
    # 1) Get sliding windows along the time axis.
    #    Result: (n_channels, n_windows, window_length)
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_length, axis=1
    )
    
    # 2) Reorder to (n_windows, n_channels, window_length)
    windows = windows.transpose(1, 0, 2)
    
    # 3) Cast to float32 and return
    return windows.astype(np.float32)


def per_timepoint_labels(window_labels, window_length):
    """
    Compute a 1-D label for each original time-sample.

    Each sample’s label is the mean of the labels of all sliding
    windows that cover that sample (stride = 1).

    Parameters
    ----------
    window_labels : 1-D array-like, shape (n_windows,)
        Label for each sliding window.
    window_length : int
        Number of samples per window.

    Returns
    -------
    sample_labels : np.ndarray, shape (n_windows + window_length - 1,)
        Label for every unique time-point, in chronological order.
        `sample_labels[t]` is the averaged label of the t-th sample
        of the original continuous signal.
    """
    window_labels = np.asarray(window_labels, dtype=float)
    n_windows = window_labels.size

    # --- denominator: how many windows overlap each time-point ---
    counts = np.convolve(
        np.ones(n_windows, dtype=int),        # 1 for every window
        np.ones(window_length, dtype=int),    # box filter of length L
        mode="full"                           # ➜ length n_windows+L-1
    )

    # --- numerator: sum of labels of windows covering each sample ---
    sums = np.convolve(
        window_labels,                        # window labels
        np.ones(window_length, dtype=float),  # same box filter
        mode="full"
    )

    # element-wise average
    return sums / counts
def add_noise_batch(data, noise_std=0.01):
    """
    Adds Gaussian noise independently to each channel of every sample.

    Args:
        data: torch.Tensor of shape (n_samples, n_time_steps, n_channels)
        noise_std: standard deviation of the Gaussian noise.
        
    Returns:
        torch.Tensor: Data with additive noise, same shape as input.
    """
    noise = torch.randn_like(data) * noise_std
    return data + noise

def scale_batch(data, scale_max=0.1):
    """
    Scales each channel in each sample by an independent random factor
    in [1-scale_max, 1+scale_max].
    """
    n_samples, T, n_channels = data.shape
    # Uniform in [1 - scale_max, 1 + scale_max]
    scales = 1.0 + (torch.rand(n_samples, n_channels, device=data.device) * 2 * scale_max - scale_max)
    scales = scales.view(n_samples, 1, n_channels)
    return data * scales

def subtract_random_value_batch(data, subtract_max):
    """
    Subtracts an independent random constant from each channel in each sample.
    The random constant is drawn uniformly from [-subtract_max, subtract_max].

    Args:
        data: torch.Tensor of shape (n_samples, n_time_steps, n_channels)
        subtract_max: maximum absolute value for the random subtraction.

    Returns:
        torch.Tensor: Data with a random constant subtracted from each channel,
                      same shape as input.
    """
    n_samples, T, n_channels = data.shape
    random_values = torch.rand(n_samples, n_channels, device=data.device) * (2 * subtract_max) - subtract_max
    random_values = random_values.view(n_samples, 1, n_channels)
    return data - random_values

def time_shift_batch(data, shift_max=10):
    """
    Applies an independent random time shift to each channel in every sample.
    The shifting is done circularly (wrap-around).

    Args:
        data: torch.Tensor of shape (n_samples, n_time_steps, n_channels)
        shift_max: maximum number of time steps to shift (in either direction).
        
    Returns:
        torch.Tensor: Time-shifted data, same shape as input.
    """
    if shift_max <= 0:
        return data
    n_samples, T, n_channels = data.shape
    # Generate a random shift for each sample and channel
    shifts = torch.randint(-shift_max, shift_max, (n_samples, 1, n_channels), device=data.device)
    # Create a time index grid of shape (n_samples, T, n_channels)
    time_idx = torch.arange(T, device=data.device).view(1, T, 1).expand(n_samples, T, n_channels)
    # Compute new indices with each channel's shift (wrap-around with modulo)
    new_indices = (time_idx - shifts) % T
    # Use torch.gather to reorder the time dimension accordingly
    return torch.gather(data, dim=1, index=new_indices)

def augment_fnirs_data(data_array, noise_std=0.005, subtract_max=0.005, scale_max=0.01, shift_max=0):
    """
    Generates augmented versions of the fNIRS data for an entire batch,
    applying random scaling, noise addition, random subtraction, and time shifting
    independently on each channel.

    Args:
        data_array: torch.Tensor of shape (n_samples, n_time_steps, n_channels)
        noise_std: standard deviation for Gaussian noise.
        subtract_max: maximum absolute value for random subtraction.
        scale_max: maximum scaling factor.
        shift_max: maximum shift in time steps.
    
    Returns:
        torch.Tensor: Augmented data, same shape as the input.
    """
    augmented = data_array.clone()  # Clone to avoid modifying original data
    augmented = scale_batch(augmented, scale_max)
    augmented = add_noise_batch(augmented, noise_std)
    augmented = subtract_random_value_batch(augmented, subtract_max)
    augmented = time_shift_batch(augmented, shift_max)
    return augmented

def test_augment_fnirs_data(X):
    X = torch.Tensor(X)
    X_trans = X.clone()
    X_trans = augment_fnirs_data(X_trans, noise_std=0.000, subtract_max=0.000, scale_max=0.00, shift_max=0)
    print(torch.allclose(X, X_trans))