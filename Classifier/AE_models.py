import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        # Second convolution: downsample by factor 2
        self.avg_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.avg_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Third convolution: further downsample by factor 2
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32,
                               kernel_size=2, stride=1, padding=0)
        
        # Compute the output length after conv layers using the formula:
        # L_out = floor((L + 2*padding - kernel_size)/stride + 1)
        def conv_out_length(L, kernel_size=3, stride=1, padding=1):
            return (L + 2 * padding - kernel_size) // stride + 1
        

        
        L1 = conv_out_length(input_length, kernel_size=3, stride=1, padding=1)
        L2 = conv_out_length(L1, kernel_size=2, stride=2, padding=0)
        L3 = conv_out_length(L2, kernel_size=3, stride=1, padding=1)
        L5 = conv_out_length(L3, kernel_size=2, stride=1, padding=0)
        self.conv_output_length = L5  # save for the decoder
        
        self.flattened_size = 32 * L5
        self.encoder_fc1 = nn.Linear(self.flattened_size, self.flattened_size//2)
        self.encoder_dropout = nn.Dropout(0.5)
        self.encoder_fc2 = nn.Linear(self.flattened_size//2, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_channels, input_length)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)     # -> (batch, 32, L)
        x = self.avg_pool1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)     # -> (batch, 32, L2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)     # -> (batch, 16, L3)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)    # flatten
        x = self.encoder_dropout(x)
        x = F.sigmoid(self.encoder_fc1(x))
        x = self.encoder_dropout(x)
        latent = self.encoder_fc2(x)           # -> (batch, latent_dim)
        return latent

# -------------------------------
# Reconstruction Decoder Module
# -------------------------------
class ReconstructionDecoder(nn.Module):
    def __init__(self, latent_dim, flattened_size, conv_output_length,
                 output_channels, original_length):
        """
        Args:
            latent_dim (int): Dimension of latent vector.
            flattened_size (int): Size from encoder (e.g., 16 * L3).
            conv_output_length (int): L3 from the encoder.
            output_channels (int): Number of channels in the output (e.g., 40).
            original_length (int): Original number of time steps (e.g., 79).
        """
        super(ReconstructionDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, flattened_size)
        self.conv_output_length = conv_output_length
        self.output_channels = output_channels
        self.original_length = original_length

        # First upsampling block
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up1 = nn.Conv1d(in_channels=16, out_channels=32,
                                  kernel_size=3, stride=1, padding=1)
        # Second upsampling block
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_up2 = nn.Conv1d(in_channels=32, out_channels=32,
                                  kernel_size=3, stride=1, padding=1)
        # Final reconstruction layer
        self.conv_final = nn.Conv1d(in_channels=32, out_channels=output_channels,
                                    kernel_size=3, stride=1, padding=1)
        # After two upsampling operations, the time dimension becomes conv_output_length * 4.
        self.final_length = conv_output_length * 4
        self.crop_amount = self.final_length - original_length
        if self.crop_amount < 0:
            raise ValueError("Upsampled length is smaller than original length.")
        self.crop = Cropping1D(crop_right=self.crop_amount)

    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.fc(z)  # -> (batch, flattened_size)
        batch_size = x.shape[0]
        # Reshape to (batch, 16, conv_output_length)
        x = x.view(batch_size, 16, self.conv_output_length)
        x = self.upsample1(x)                   # -> (batch, 16, conv_output_length*2)
        x = F.relu(self.conv_up1(x))            # -> (batch, 32, conv_output_length*2)
        x = self.upsample2(x)                   # -> (batch, 32, conv_output_length*4)
        x = F.relu(self.conv_up2(x))            # -> (batch, 32, conv_output_length*4)
        x = self.conv_final(x)                  # -> (batch, output_channels, conv_output_length*4)
        # Crop to original length if necessary
        if self.crop_amount > 0:
            x = self.crop(x)
        return x

# -------------------------------
# Reconstruction Autoencoder
# -------------------------------
class ReconstructionAutoencoder(nn.Module):
    def __init__(self, input_channels, input_length, latent_dim):
        """
        Combines the encoder and reconstruction decoder.
        """
        super(ReconstructionAutoencoder, self).__init__()
        self.encoder = Encoder(input_channels, input_length, latent_dim)
        self.decoder = ReconstructionDecoder(
            latent_dim=latent_dim,
            flattened_size=self.encoder.flattened_size,
            conv_output_length=self.encoder.conv_output_length,
            output_channels=input_channels,
            original_length=input_length
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
    
    def get_latent(self, x):
        return self.encoder(x)

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
# Training Functions
# -------------------------------
def train_reconstruction_autoencoder(model, train_loader, val_loader, epochs, device, verbose=False):
    """
    Training loop for the reconstruction autoencoder.
    """
    criterion = nn.MSELoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)  # input and target are the same
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if verbose: 
            print(f"Reconstruction AE Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    return history

def train_classification_autoencoder(model, train_loader, val_loader, epochs, device, verbose=False):
    """
    Training loop for the classification autoencoder.
    """
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
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

def create_sliding_windows_old(data, labels, window_length):
    """
    Splits each epoch in 'data' into overlapping windows using a sliding window approach,
    and replicates the corresponding label for each new window.
    
    Parameters:
        data (np.ndarray): Input array with shape (n_epochs, n_channels, n_times)
        labels (np.ndarray or list): Array or list of labels with length n_epochs.
        window_length (int): Number of observations (time points) per window.
        
    Returns:
        windows (np.ndarray): Array with shape (n_epochs_new, n_channels, window_length)
                              where n_epochs_new = n_epochs * (n_times - window_length + 1).
        new_labels (np.ndarray): Array with shape (n_epochs_new,) where each window
                                 is assigned the label of its original epoch.
    """
    # Create sliding windows along the time axis.
    # This returns an array of shape (n_epochs, n_channels, n_windows, window_length)
    windows = np.lib.stride_tricks.sliding_window_view(data, window_length, axis=2)
    
    # Transpose to get shape (n_epochs, n_windows, n_channels, window_length)
    windows = windows.transpose(0, 2, 1, 3)
    
    # Reshape to combine the epochs and windows dimensions:
    n_epochs, n_windows, n_channels, _ = windows.shape
    windows = windows.reshape(n_epochs * n_windows, n_channels, window_length)
    start_cut_off = n_windows//5
    end_cut_off = n_windows//8
    new_labels  =[]
    for label in labels:
        new_labels += [0] * start_cut_off + [label] * (n_windows - start_cut_off - end_cut_off) + [0] * (end_cut_off)
    # Repeat each label n_windows times to correspond with each window created from an epoch.
    new_labels = np.array(new_labels)
    
    return windows, new_labels


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
            For each epoch, the first (n_windows // 5) and the last (n_windows // 8) windows get label 0,
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
        end_cut = n_windows // 8
        
        # For each epoch in the current array, create a label vector for its windows.
        for i in label:
            new_labels.extend([0] * start_cut + [i] * (n_windows - start_cut - end_cut) + [0] * (end_cut))
        
    # Concatenate windows from all arrays into a single array.
    windows_all = np.concatenate(windows_all, axis=0)
    new_labels = np.array(new_labels)  # This is a 1D array.
    
    return windows_all.astype(np.float32), new_labels