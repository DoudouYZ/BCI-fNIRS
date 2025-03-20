import os
import sys
from AE_models import ReconstructionAutoencoder, ClassificationAutoencoder, train_reconstruction_autoencoder, train_classification_autoencoder
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import your preprocessing pipeline (make sure Preprocessing.py is in your PYTHONPATH)
from Preprocessing import get_epochs
# Import our AE models and training functions


# -------------------------
# Data Preparation
# -------------------------
# Load preprocessed epochs
epochs = get_epochs()

# Extract data in a given time window (e.g., 0 to 10 seconds)
# MNE's get_data() returns an array of shape (n_epochs, n_channels, n_times)
data = epochs.copy().crop(tmin=0, tmax=10).get_data()
# For PyTorch, we want the input shape to be (batch, channels, time)
X = data.astype(np.float32)  # shape: (n_epochs, n_channels, n_times)

# Extract labels (assumed to be in epochs.events[:, -1])
# Our labels are assumed to be: 1: Control, 2: Tapping/Left, 3: Tapping/Right.
# To use CrossEntropyLoss, we want labels as LongTensor with values 0,1,2.
labels = epochs.events[:, -1].astype(np.int64) - 1  # subtract one to have classes 0,1,2

# Split into train and test sets
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, labels, test_size=0.2, random_state=42)

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

# -------------------------
# Hyperparameters
# -------------------------
# X has shape: (n_epochs, n_channels, n_times)
input_channels = X_train.shape[1]  # e.g., 40
input_length = X_train.shape[2]    # e.g., 79 (depends on crop window)
latent_dim = 2                     # for a 2D latent space visualization
num_classes = 3
epochs_num = 50

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Reconstruction Autoencoder
# -------------------------
print("Training Reconstruction Autoencoder...")
recon_ae = ReconstructionAutoencoder(input_channels, input_length, latent_dim)
history_recon = train_reconstruction_autoencoder(recon_ae, train_loader_recon, val_loader_recon, epochs_num, device)

# Extract latent representations on test set
recon_ae.eval()
with torch.no_grad():
    latent_recon = recon_ae.get_latent(X_test.to(device)).cpu().numpy()

# Plot the 2D latent representation (if latent_dim == 2)
plt.figure(figsize=(6, 5))
plt.scatter(latent_recon[:, 0], latent_recon[:, 1],
            c=y_test_np, cmap='viridis', s=50, alpha=0.7)
plt.title('Latent Representation - Reconstruction AE')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar(label='Class')
plt.show()

# -------------------------
# Classification Autoencoder
# -------------------------
print("Training Classification Autoencoder...")
class_ae = ClassificationAutoencoder(input_channels, input_length, latent_dim, num_classes)
history_class = train_classification_autoencoder(class_ae, train_loader_class, val_loader_class, epochs_num, device)

# Extract latent representations from the classification AE
class_ae.eval()
with torch.no_grad():
    latent_class = class_ae.get_latent(X_test.to(device)).cpu().numpy()

plt.figure(figsize=(6, 5))
plt.scatter(latent_class[:, 0], latent_class[:, 1],
            c=y_test_np, cmap='viridis', s=50, alpha=0.7)
plt.title('Latent Representation - Classification AE')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar(label='Class')
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
