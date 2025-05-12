import itertools
import numpy as np
import random
import torch
import os
import sys
from Classifier.AE_models import ReconstructionAutoencoder, ClassificationAutoencoder, train_reconstruction_autoencoder, train_classification_autoencoder, create_sliding_windows
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
from Preprocessing import get_group_epochs_subtracting_short, get_group_epochs
import random
from tqdm import tqdm

verbose = False
def run_AE(device, test_participant, latent_dim = 2, num_classes = 3, epochs_num = 25, verbose=True, transform_data=True, noise_std=0.005, subtract_max=0.001, scale_max=0.01, shift_max=1):

    # -------------------------
    # Data Preparation
    # -------------------------
    # Load preprocessed epochs
    epochs = get_group_epochs(tmin=-5, add_hbr=False, hbr_multiplier=5.0, hbr_shift=1.0)


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
            X[:, ch, :] = (X[:, ch, :]-channel_mean) / channel_std


    # Extract labels (assumed to be in epochs.events[:, -1])
    # Our labels are assumed to be: 1: Control, 2: Tapping/Left, 3: Tapping/Right.
    # To use CrossEntropyLoss, we want labels as LongTensor with values 0,1,2.
    labels = [epoch.events[:, -1].astype(np.int64) - 1 for epoch in epochs]  # subtract one to have classes 0,1,2

    participants = [[i] * len(participant_data) for participant_data in data]

    # Split into train and test sets
    X_train, X_test, y_train, y_test =  data[:test_participant] + data[test_participant+1:], data[test_participant], labels[:test_participant] + labels[test_participant+1:], labels[test_participant]

    X_train_np, y_train_np = create_sliding_windows(X_train, y_train, 32)
    X_test_np, y_test_np = create_sliding_windows([X_test], [y_test], 32)
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
    


    # -------------------------
    # Classification Autoencoder
    # -------------------------
    if verbose:
        print("Training Classification Autoencoder...")
    class_ae = ClassificationAutoencoder(input_channels, input_length, latent_dim, num_classes)
    history_class = train_classification_autoencoder(class_ae, train_loader_class, val_loader_class, epochs_num, device, verbose=verbose, transform_data=transform_data, noise_std=noise_std, subtract_max=subtract_max, scale_max=scale_max, shift_max=shift_max)

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

# Define your hyperparameter lists:
# noise_std_list = [0.01, 0.005, 0.002]
# subtract_max_list = [0.01, 0.005, 0.001]
# scale_max_list = [0.01, 0.005]
# shift_max_list = [0, 1, 2]
noise_std_list = [0.005, 0.01]
subtract_max_list = [0.01, 0.005]
scale_max_list = [0.01, 0.005]
shift_max_list = [0, 1]

# List of seeds to try; these will simulate different data splits or initialization randomness.
seed_list = [42, 43, 44]

# Container to store the results for each hyperparameter combination
results = []

# Variable to store the best parameters and best average accuracy
best_accuracy = 0.0
best_params = None

# Loop over all combinations using itertools.product:
for noise_std, subtract_max, scale_max, shift_max in tqdm(itertools.product(
        noise_std_list, subtract_max_list, scale_max_list, shift_max_list), desc="Testing combinations of params"):
    accuracies = []  # List to collect accuracy over multiple seeds for this combination
    
    # Loop over different seeds (or different folds)
    for seed in seed_list:
        # Set the seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        for i in range(5):    
            # Now run your AE training routine.
            # Make sure your run_AE function has been updated to accept these extra parameters.
            latent_class_test, y_test_np, latent_class_train, y_train_np, accuracy = run_AE(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                test_participant=i,
                epochs_num=30,
                verbose=verbose,
                num_classes=3,
                latent_dim=3,
                transform_data=True,
                noise_std=noise_std,          # <-- New parameter
                subtract_max=subtract_max,    # <-- New parameter
                scale_max=scale_max,          # <-- New parameter
                shift_max=shift_max           # <-- New parameter
            )
            accuracies.append(accuracy)
        
    # Compute the average accuracy for this hyperparameter combination
    avg_accuracy = np.mean(accuracies)
    results.append({
        'noise_std': noise_std,
        'subtract_max': subtract_max,
        'scale_max': scale_max,
        'shift_max': shift_max,
        'average_accuracy': avg_accuracy
    })
    
    if verbose: 
        print(f"Combination: noise_std={noise_std}, subtract_max={subtract_max}, "
          f"scale_max={scale_max}, shift_max={shift_max} --> Avg Accuracy: {avg_accuracy:.4f}")
    
    # Update best hyperparameter combination if needed
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_params = (noise_std, subtract_max, scale_max, shift_max)

print("\nBest hyperparameter combination:")
print(f"noise_std = {best_params[0]}, subtract_max = {best_params[1]}, "
      f"scale_max = {best_params[2]}, shift_max = {best_params[3]} with average accuracy = {best_accuracy:.4f}")
