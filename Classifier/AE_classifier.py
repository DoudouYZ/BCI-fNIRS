import os
import sys
from AE_models import ReconstructionAutoencoder, ClassificationAutoencoder, train_reconstruction_autoencoder, train_classification_autoencoder, create_sliding_windows
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import your preprocessing pipeline (make sure Preprocessing.py is in your PYTHONPATH)
from Preprocessing import get_group_epochs, multiply_hbr_in_epochs
import random
seed = 43
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Data Preparation
# -------------------------
# Load preprocessed epochs
epochs = get_group_epochs(add_hbr=False, hbr_multiplier=5.0, hbr_shift=1.0, tmin=-5, tmax=15, force_download=False)

# Keep only the HbO channels by selecting channels whose names contain "hbo"
# for i in range(len(epochs)):
#     hbo_names = [ch for ch in epochs[i].ch_names if 'hbo' in ch.lower()]
#     epochs[i].pick(hbo_names)

# epochs = multiply_hbr_in_epochs(epochs, factor=5.0, boundary=3*10**(-6))

# Extract data in a given time window (e.g., 0 to 10 seconds)
# MNE's get_data() returns an array of shape (n_epochs, n_channels, n_times)
data = []
for i in range(len(epochs)):
    data.append(epochs[i].copy().crop(tmin=0, tmax=14).get_data())
# For PyTorch, we want the input shape to be (batch, channels, time)

for X in data:
    n_epochs, n_channels, n_times = X.shape

    for ch in range(n_channels):
        channel_mean = np.mean(X[:, ch, :])
        channel_std = np.std(X[:, ch, :])
        X[:, ch, :] = (X[:, ch, :] - channel_mean) / channel_std


# Extract labels (assumed to be in epochs.events[:, -1])
# Our labels are assumed to be: 1: Control, 2: Tapping/Left, 3: Tapping/Right.
# To use CrossEntropyLoss, we want labels as LongTensor with values 0,1,2.
labels = [epoch.events[:, -1].astype(np.int64) - 1 for epoch in epochs]  # subtract one to have classes 0,1,2

participants = [[i] * len(participant_data) for participant_data in data]

test_idx = 4 # Test subject

# -------------------------
# Hyperparameters
# -------------------------
# X has shape: (n_epochs, n_channels, n_times)
latent_dim = 2                     # for a 2D latent space visualization
num_classes = 3
epochs_num = 4
window_length = 20

reconstruct = False
classify = True

# Split into train and test sets
X_train, X_test, y_train, y_test =  data[:test_idx] + data[test_idx+1:], data[test_idx], labels[:test_idx] + labels[test_idx+1:], labels[test_idx]

print(len(X_train))
quit

X_train_np, y_train_np = create_sliding_windows(X_train, y_train, window_length=window_length)
X_test_np, y_test_np = create_sliding_windows([X_test], [y_test], window_length=window_length)
# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train_np)
X_test = torch.from_numpy(X_test_np)
y_train = torch.from_numpy(y_train_np)
y_test = torch.from_numpy(y_test_np)

# Create TensorDatasets and DataLoaders
batch_size = 16
train_dataset_recon = TensorDataset(X_train)  # for reconstruction, input=target
val_dataset_recon = TensorDataset(X_test)
train_loader_recon = DataLoader(train_dataset_recon, batch_size=batch_size, shuffle=True)
val_loader_recon = DataLoader(val_dataset_recon, batch_size=batch_size, shuffle=False)

train_dataset_class = TensorDataset(X_train, y_train)
val_dataset_class = TensorDataset(X_test, y_test)
train_loader_class = DataLoader(train_dataset_class, batch_size=batch_size, shuffle=True)
val_loader_class = DataLoader(val_dataset_class, batch_size=batch_size, shuffle=False)

input_channels = X_train.shape[1]  # e.g., 40
input_length = X_train.shape[2]    # e.g., 79 (depends on crop window)


if classify:
    # -------------------------
    # Classification Autoencoder
    # -------------------------
    print("Training Classification Autoencoder...")
    class_ae = ClassificationAutoencoder(input_channels, input_length, latent_dim, num_classes)
    history_class = train_classification_autoencoder(class_ae, train_loader_class, val_loader_class, epochs_num, device, verbose=True)

    # Extract latent representations from the classification AE
    class_ae.eval()
    with torch.no_grad():
        latent_class = class_ae.get_latent(X_test.to(device)).cpu().numpy()

    plt.figure(figsize=(7, 5))
    plt.scatter(latent_class[:, 0], latent_class[:, 1],
                c=y_test_np, cmap='viridis', s=50, alpha=0.7)
    plt.title('Latent Representation - Classification AE')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    legend_labels = {0: "Control", 1: "Tapping/Left", 2: "Tapping/Right"}
    handles = [mpatches.Patch(color=plt.cm.viridis(i/2.0), label=legend_labels[i]) for i in sorted(legend_labels.keys())]
    plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Evaluation Metrics for Classification AE
    # -------------------------
    class_ae.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader_class:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = class_ae(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    print("Classification AE Test Accuracy: {:.2f}%".format(accuracy * 100))
