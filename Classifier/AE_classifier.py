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
from Preprocessing import get_group_epochs
import random
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_AE(device, test_participant, type = "classify", latent_dim = 2, num_classes = 3, epochs_num = 25, verbose=True):

    # -------------------------
    # Data Preparation
    # -------------------------
    # Load preprocessed epochs
    epochs = get_group_epochs(tmin=-0.2)


    # Extract data in a given time window (e.g., 0 to 10 seconds)
    # MNE's get_data() returns an array of shape (n_epochs, n_channels, n_times)
    data = []
    for i in range(len(epochs)):
        data.append(epochs[i].copy().crop(tmin=0, tmax=10).get_data())
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

    # Split into train and test sets
    X_train, X_test, y_train, y_test =  data[:test_participant] + data[test_participant+1:], data[test_participant], labels[:test_participant] + labels[test_participant+1:], labels[test_participant]

    X_train_np, y_train_np = create_sliding_windows(X_train, y_train, 20)
    X_test_np, y_test_np = create_sliding_windows([X_test], [y_test], 20)
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
    input_channels = X_train.shape[1]
    input_length = X_train.shape[2]    
    

    if type == "classify":
        # -------------------------
        # Classification Autoencoder
        # -------------------------
        print("Training Classification Autoencoder...")
        class_ae = ClassificationAutoencoder(input_channels, input_length, latent_dim, num_classes)
        history_class = train_classification_autoencoder(class_ae, train_loader_class, val_loader_class, epochs_num, device, verbose=verbose)

        # Extract latent representations from the classification AE
        class_ae.eval()
        with torch.no_grad():
            latent_class_test = class_ae.get_latent(X_test.to(device)).cpu().numpy()
            latent_class_train = class_ae.get_latent(X_train.to(device)).cpu().numpy()

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
        return latent_class_test, y_test_np, latent_class_train, y_train_np, accuracy



latent_class_test, y_test_np, latent_class_train, y_train_np, accuracy = run_AE(device=device, test_participant=0, epochs_num=20, verbose=True)
plt.figure(figsize=(7, 5))
plt.scatter(latent_class_test[:, 0], latent_class_test[:, 1],
            c=y_test_np, cmap='viridis', s=15, alpha=0.7)
plt.title(f'Latent Representation - AE test (acc: {round(accuracy, 2)})')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
legend_labels = {0: "Control", 1: "Tapping/Left", 2: "Tapping/Right"}
handles = [mpatches.Patch(color=plt.cm.viridis(i/2.0), label=legend_labels[i]) for i in sorted(legend_labels.keys())]
plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.show()
plt.figure(figsize=(7, 5))
plt.scatter(latent_class_train[:, 0], latent_class_train[:, 1],
            c=y_train_np, cmap='viridis', s=8, alpha=0.7)
plt.title(f'Latent Representation - AE ')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
legend_labels = {0: "Control", 1: "Tapping/Left", 2: "Tapping/Right"}
handles = [mpatches.Patch(color=plt.cm.viridis(i/2.0), label=legend_labels[i]) for i in sorted(legend_labels.keys())]
plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 0.5))
plt.tight_layout()
plt.show()