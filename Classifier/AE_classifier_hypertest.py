import os
import sys
from AE_models import ReconstructionAutoencoder, ClassificationAutoencoder, train_reconstruction_autoencoder, train_classification_autoencoder, create_sliding_windows
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from itertools import product
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from AE_models import ClassificationAutoencoder, train_classification_autoencoder, create_sliding_windows

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import TensorDataset, DataLoader
import random

# Import your preprocessing pipeline 
from Preprocessing import get_group_epochs

# -------------------------
# Hyperparameter grid
# -------------------------
test_idx_list = [0, 1, 2, 3, 4]
hbr_multiplier_list = [1, 3, 5]
hbr_shift_list = [0, 1, 2, 3]
window_length_list = [15, 20, 30]
add_hbr_list = [True, False]
seed_list = [42, 43, 44]
epochs_num = 5  # training epochs for AE (set as needed)
tmin, tmax = -5, 15

# Container for all runs
results = []

# Count total runs for tqdm
total_runs = (
    len(seed_list) *
    len(test_idx_list) *
    len(hbr_multiplier_list) *
    len(hbr_shift_list) *
    len(window_length_list) *
    len(add_hbr_list)
)

# Main loop (verbose=False everywhere)
pbar = tqdm(total=total_runs, desc="Running experiments")
for seed, test_idx, hbr_multiplier, hbr_shift, window_length, add_hbr in product(
    seed_list, test_idx_list, hbr_multiplier_list, hbr_shift_list, window_length_list, add_hbr_list
):
    # Set the random seed for reproducibility for each run.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Load preprocessed epochs using current hyperparameters for HbR handling.
    epochs = get_group_epochs(add_hbr=add_hbr, hbr_multiplier=hbr_multiplier, hbr_shift=hbr_shift, tmin=tmin, tmax=tmax, force_download=False)
    
    # Extract data in a given time window (e.g., 0 to 14 seconds)
    data = []
    for i in range(len(epochs)):
        data.append(epochs[i].copy().crop(tmin=0, tmax=14).get_data())
    
    # Normalize each channel for every trial
    for X in data:
        n_epochs, n_channels, n_times = X.shape
        for ch in range(n_channels):
            channel_mean = np.mean(X[:, ch, :])
            channel_std = np.std(X[:, ch, :])
            X[:, ch, :] = (X[:, ch, :] - channel_mean) / channel_std
            
    # Extract labels (expected to be in epochs.events[:, -1] with classes 1,2,3; subtract one to have 0,1,2)
    labels = [epoch.events[:, -1].astype(np.int64) - 1 for epoch in epochs]
    
    # Split into train and test sets based on test_idx
    X_train = data[:test_idx] + data[test_idx+1:]
    y_train = labels[:test_idx] + labels[test_idx+1:]
    X_test = data[test_idx]
    y_test = labels[test_idx]
    
    # Convert sliding windows for both train and test sets.
    X_train_np, y_train_np = create_sliding_windows(X_train, y_train, window_length=window_length)
    X_test_np, y_test_np = create_sliding_windows([X_test], [y_test], window_length=window_length)
    
    # Convert to torch tensors
    X_train_tensor = torch.from_numpy(X_train_np)
    X_test_tensor = torch.from_numpy(X_test_np)
    y_train_tensor = torch.from_numpy(y_train_np)
    y_test_tensor = torch.from_numpy(y_test_np)
    
    # Create DataLoaders
    batch_size = 16
    train_dataset_class = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset_class = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader_class = DataLoader(train_dataset_class, batch_size=batch_size, shuffle=True)
    val_loader_class = DataLoader(val_dataset_class, batch_size=batch_size, shuffle=False)
    
    # Determine input_channels and input_length from X_train_tensor shape
    input_channels = X_train_tensor.shape[1]
    input_length = X_train_tensor.shape[2]
    latent_dim = 2
    num_classes = 3
    
    # Build classification AE and train (set verbose=False)
    class_ae = ClassificationAutoencoder(input_channels, input_length, latent_dim, num_classes)
    history_class = train_classification_autoencoder(class_ae, train_loader_class, val_loader_class, epochs_num, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=False)
    
    # Evaluate model on validation set
    class_ae.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader_class:
            inputs = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            targets = targets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            logits = class_ae(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total if total > 0 else 0.0
    
    # Save result
    results.append({
        'seed': seed,
        'test_idx': test_idx,
        'hbr_multiplier': hbr_multiplier,
        'hbr_shift': hbr_shift,
        'window_length': window_length,
        'add_hbr': add_hbr,
        'accuracy': accuracy
    })
    
    pbar.update(1)

pbar.close()

# --------------------------------------
# Aggregation and reporting of results
# --------------------------------------
# Average results over seeds for identical hyperparameter settings (excluding seed)
avg_results = defaultdict(list)
for res in results:
    key = (res['test_idx'], res['hbr_multiplier'], res['hbr_shift'], res['window_length'], res['add_hbr'])
    avg_results[key].append(res['accuracy'])

avg_results_mean = {key: np.mean(acc_list) for key, acc_list in avg_results.items()}

# Find the top 5 hyperparameter sets overall (based on average accuracy)
sorted_results = sorted(avg_results_mean.items(), key=lambda x: x[1], reverse=True)
top_5 = sorted_results[:5]

print("\nTop 5 hyperparameter configurations (averaged over seeds):")
for key, acc in top_5:
    test_idx, hbr_multiplier, hbr_shift, window_length, add_hbr = key
    print(f"test_idx={test_idx}, hbr_multiplier={hbr_multiplier}, hbr_shift={hbr_shift}, window_length={window_length}, add_hbr={add_hbr} -> Average Val Accuracy: {acc*100:.2f}%")

# Determine which test_idx has the highest overall average validation accuracy
test_idx_acc = defaultdict(list)
for key, acc in avg_results_mean.items():
    test_idx, *_ = key
    test_idx_acc[test_idx].append(acc)
avg_test_idx = {k: np.mean(v) for k, v in test_idx_acc.items()}
best_test_idx = max(avg_test_idx.items(), key=lambda x: x[1])[0]

print(f"\nTest subject index with highest average validation accuracy: {best_test_idx}")

# Print full summary of all results (or you can save this to a file)
print("\nAll experiment results (each line is one run):")
for res in results:
    print(res)

# Save all results to a CSV file
df = pd.DataFrame(results)
csv_filepath = os.path.join(os.path.dirname(__file__), "hyper_results.csv")
df.to_csv(csv_filepath, index=False)
print(f"\nAll experiment results have been saved to: {csv_filepath}")