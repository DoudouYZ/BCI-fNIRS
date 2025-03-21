import os
import sys
from AE_models import create_sliding_windows
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from Preprocessing import get_epochs
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import your preprocessing pipeline (make sure Preprocessing.py is in your PYTHONPATH)
from Preprocessing import get_epochs
import random

plot = True

# -------------------------
# Data Preparation
# -------------------------
# Load preprocessed epochs
epochs = get_epochs()

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# Extract data in a given time window (e.g., 0 to 10 seconds)
# MNE's get_data() returns an array of shape (n_epochs, n_channels, n_times)
data = epochs.copy().crop(tmin=0, tmax=10).get_data()

# For PyTorch, we want the input shape to be (batch, channels, time)
X = data.astype(np.float32)  # shape: (n_epochs, n_channels, n_times)
labels = epochs.events[:, -1]  # getting labels from epoch events

n_epochs, n_channels, n_times = X.shape

for ch in range(n_channels):
    channel_mean = np.mean(X[:, ch, :])
    channel_std = np.std(X[:, ch, :])
    X[:, ch, :] = (X[:, ch, :] - channel_mean) / channel_std


# Extract labels (assumed to be in epochs.events[:, -1])
# Our labels are assumed to be: 1: Control, 2: Tapping/Left, 3: Tapping/Right.
# To use CrossEntropyLoss, we want labels as LongTensor with values 0,1,2.
labels = epochs.events[:, -1].astype(np.int64) - 1  # subtract one to have classes 0,1,2

# Split into train and test sets

X_train_np, y_train_np = create_sliding_windows(X, labels, 1)
n_epochs, n_channels, window_length = X_train_np.shape
X_train_np = X_train_np.reshape(n_epochs, n_channels)


# Set the number of PCA components (adjust this as needed)
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_train_np)
if plot:
    plt.figure(figsize=(7, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1],
                c=y_train_np, cmap='viridis', s=50, alpha=0.7)
    plt.title('Latent Representation - Classification AE')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    legend_labels = {0: "Control", 1: "Tapping/Left", 2: "Tapping/Right"}
    handles = [mpatches.Patch(color=plt.cm.viridis(i/2.0), label=legend_labels[i]) for i in sorted(legend_labels.keys())]
    plt.legend(handles=handles, title="Class", bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()
