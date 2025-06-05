import os
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from preprocessing_mne import get_raw_subject_data
from ICA_plotting import *
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def reshape_epochs_with_indices(epochs, n_epochs_to_include):
    """
    Reshape epochs data to a 2D array and create an index array that indicates the original epoch for each sample.
    Using the minimum number of samples from the conditions for uniformity for concatenation.

    Parameters:
        epochs: MNE epochs object containing the data.
        n_epochs_to_include: Number of epochs to include (using the minimum sample count for uniformity).

    Returns:
        tuple: (reshaped data, array of original epoch indices)
    """
    epoch_data = epochs.get_data()[:n_epochs_to_include, :, :]
    n_epoch, n_channels, n_epoch_size = epoch_data.shape
    epoch_data_reshaped = epoch_data.reshape(n_channels, n_epoch * n_epoch_size).T
    orig_indices = np.repeat(epochs.selection[:n_epochs_to_include], n_epoch_size)
    return epoch_data_reshaped, orig_indices

def load_data(subject=3):
    """
    Load data from MNE, reshape epochs, and concatenate data from control, left, and right conditions.
    We assume that the clinical trials can split the data up into control and potential activation

    Parameters:
        subject: Subject ID to load data for.

    Returns:
        tuple: (X, orig_indices, y, control_data, left_data, right_data)
            - X: concatenated continuous data.
            - orig_indices: epoch indices for each sample.
            - y: labels for each sample.
            - control_data, left_data, right_data: individual condition data.
    """
    epochs = get_raw_subject_data(subject=subject)
    control = epochs['Control']
    left = epochs['Tapping_Left']
    right = epochs['Tapping_Right']
    
    min_bound = np.min([x.get_data().shape[0] for x in [control, left, right]])
    
    control_data, ctrl_idx = reshape_epochs_with_indices(control, min_bound)
    left_data, left_idx = reshape_epochs_with_indices(left, min_bound)
    right_data, right_idx = reshape_epochs_with_indices(right, min_bound)
    
    # Concatenating data. Doesn't impact ICA
    X = np.concatenate([control_data, left_data, right_data], axis=0)
    orig_indices = np.concatenate([ctrl_idx, left_idx, right_idx])
    y = np.concatenate([np.full(control_data.shape[0], 1),
                        np.full(left_data.shape[0], 2),
                        np.full(right_data.shape[0], 3)])
    return X, orig_indices, y, control_data, left_data, right_data

def run_ica(X, n_components=5, standardized=False):
    """
    Run Independent Component Analysis (ICA) to unmix the data.

    Parameters:
        X: Input data array.
        n_components: Number of independent components to retrieve.

    Returns:
        X_ica: Transformed data with independent components.
    """
    if standardized:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = X_scaled
    ica = FastICA(n_components=n_components, max_iter=1000, tol=0.0001, random_state=42)
    X_ica = ica.fit_transform(X)
    return X_ica

def compute_sliding_windows(X_ica, y, window_size, step_size):
    """
    Compute features by calculating the mean over sliding windows for each class.
    Windows from different classes don't overlap.

    Parameters:
        X_ica: Data after ICA transformation.
        y: Labels for each sample.
        window_size: Number of samples in each sliding window.
        step_size: Step size between sliding windows.

    Returns:
        tuple: (features, labels) for each sliding window.
    """
    features_list, labels_list = [], []
    for cl in np.unique(y):
        class_mask = (y == cl)
        class_data = X_ica[class_mask, :]
        n_samples_class = class_data.shape[0]
        for start in range(0, n_samples_class - window_size + 1, step_size):
            window = class_data[start:start+window_size, :]
            avg_feature = np.mean(window, axis=0)
            features_list.append(avg_feature)
            labels_list.append(cl)
    return np.array(features_list), np.array(labels_list)

def run_gmm(features, n_components=3):
    """
    Cluster sliding window features using Gaussian Mixture Model.

    Parameters:
        features: Feature matrix extracted from sliding windows.
        n_components: Number of GMM components (clusters).

    Returns:
        tuple: (gmm model, cluster labels for each feature)
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(features)
    cluster_labels = gmm.predict(features)
    return gmm, cluster_labels

# ------------------ Main Flow ------------------
if __name__ == '__main__':
    window_size = 300
    step_size = 30

    # Load data and concatenate epochs from control, left, and right conditions
    X, orig_indices, y, control_data, left_data, right_data = load_data(subject=3)
    
    # Run ICA to obtain independent components
    X_ica = run_ica(X, n_components=5, standardized=False)
    
    # Plot the independent components
    plot_ica_components(X_ica)
    
    # Calculate unsorted boundaries: order: control, left, right
    boundary0 = 0
    boundary1 = control_data.shape[0]
    boundary2 = boundary1 + left_data.shape[0]
    boundary3 = boundary2 + right_data.shape[0]
    boundaries = [boundary0, boundary1, boundary2, boundary3]
    X_ica_unsorted = X_ica  # Preserved unsorted order

    # Plot a single control epoch with sliding windows.
    unique_epochs = np.unique(orig_indices)
    np.random.shuffle(unique_epochs)
    control_epoch_data = None
    for epoch_id in unique_epochs:
        ep_mask = orig_indices == epoch_id
        if y[ep_mask][0] == 1:
            control_epoch_data = X_ica[ep_mask, 0]
            break
    if control_epoch_data is not None:
        plot_ica_epoch(control_epoch_data, window_size)
    
    # Compute sliding window features
    features, sw_labels = compute_sliding_windows(X_ica, y, window_size, step_size)
    
    # Cluster sliding window features
    gmm, cluster_labels = run_gmm(features, n_components=3)
    
    # Plot sliding window features and their GMM clusters
    plot_scatter_features(features, sw_labels)
    plot_gmm_clusters(features, cluster_labels)
    
    # Plot the concatenated control IC1 time series
    control_mask = y == 1
    control_ic1 = X_ica[control_mask, 0]
    plot_concatenated_timeseries(control_ic1, window_size)
    
    # Plot unsorted time series with sliding window bands color-coded by GMM clusters
    block_colors = {1: "C0", 2: "C1", 3: "C2"}
    cluster_colors = {0: "pink", 1: "lightgreen", 2: "lightblue"}
    plot_unsorted_timeseries_with_clusters(X_ica_unsorted, boundaries, block_colors,
                                           window_size, step_size, sw_labels, cluster_labels, cluster_colors)