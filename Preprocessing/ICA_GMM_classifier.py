import os
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from preprocessing_mne import get_raw_subject_data, get_raw_control_subject_data
from ICA_plotting import *
from sklearn.metrics import silhouette_score
import random
import matplotlib.pyplot as plt

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
    # random_epochs = random.sample(range(n_epoch), 5)
    # for i in random_epochs:
    #     print(f"Length of epoch {i}: {epoch_data[i].shape[1]}")
    # epoch_data_reshaped = epoch_data.swapaxes(1,2).reshape(n_epoch*n_epoch_size, n_channels)
    epoch_data_reshaped = epoch_data.reshape(n_channels, n_epoch * n_epoch_size).T
    orig_indices = np.repeat(epochs.selection[:n_epochs_to_include], n_epoch_size)
    return epoch_data_reshaped, orig_indices

def load_data(subject=3, control_only=True):
    """
    Load data from MNE, reshape epochs, and concatenate data from control, left, and right conditions.
    We assume that the clinical trials can split the data up into control and potential activation

    Parameters:
        subject: Subject ID to load data for.

    Returns:
        tuple: (X, orig_indices, y, control_data, left_data, right_data, numb_samples)
            - X: concatenated continuous data.
            - orig_indices: epoch indices for each sample.
            - y: labels for each sample.
            - control_data, left_data, right_data: individual condition data.
            - numb_samples: length of the first epoch sample (number of time points).
    """

    if not control_only:
        epochs = get_raw_subject_data(subject=subject, tmin=-5, tmax=15)
        # epochs["Control"].plot_image(
        #     combine="mean",
        #     vmin=-30,
        #     vmax=30,
        #     ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
        # )
        control = epochs['Control']
        left = epochs['Tapping_Left']
        right = epochs['Tapping_Right']
        
        min_bound = np.min([x.get_data().shape[0] for x in [control, left, right]])
        
        control_only_data, ctrl_idx = reshape_epochs_with_indices(control, min_bound)
        left_data, left_idx = reshape_epochs_with_indices(left, min_bound)
        right_data, right_idx = reshape_epochs_with_indices(right, min_bound)
        
        # Concatenating data. Doesn't impact ICA
        X = np.concatenate([control_only_data, left_data, right_data], axis=0)
        orig_indices = np.concatenate([ctrl_idx, left_idx, right_idx])
        y = np.concatenate([np.full(control_only_data.shape[0], 1),
                            np.full(left_data.shape[0], 2),
                            np.full(right_data.shape[0], 3)])
        # Get number of samples from the first epoch of the control condition.
        numb_samples = control.get_data()[0].shape[1]
        return X, orig_indices, y, control_only_data, left_data, right_data, numb_samples
    else:
        control = get_raw_control_subject_data(subject=subject)
        control_only_data, ctrl_idx = reshape_epochs_with_indices(control, control.selection.shape[0])
        y = np.full(control_only_data.shape[0], 1)  # All control data labeled as 1
        # Get number of samples from the first epoch of the control condition.
        numb_samples = control.get_data()[0].shape[1]
        return control_only_data, ctrl_idx, y, control_only_data, None, None, numb_samples


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

def plot_epochs_in_original_order(X_ica, orig_indices, y, component=0):
    """
    Plot the epochs concatenated in the original time series order on a single plot.
    Each epoch is plotted sequentially, color coded by its original label:
        1: control, 2: left, 3: right.
    The legend shows one entry per label.
    """
    # Define colors and label names for each original label
    colors = {1: "C0", 2: "C1", 3: "C2"}
    label_names = {1: "control", 2: "left", 3: "right"}

    unique_epochs = np.unique(orig_indices)
    unique_epochs.sort()  # ensure original order

    plt.figure(figsize=(12, 6))
    offset = 0
    # Dictionary to hold one legend handle per label.
    legend_handles = {}
    for epoch in unique_epochs:
        mask = orig_indices == epoch
        epoch_data = X_ica[mask, component]
        # Use the first sample's label for this epoch.
        epoch_label = y[mask][0]
        epoch_color = colors.get(epoch_label, "black")
        n_samples = len(epoch_data)
        x_vals = np.arange(offset, offset + n_samples)
        plt.plot(x_vals, epoch_data, color=epoch_color)
        # Create a legend handle if it doesn't exist yet.
        if epoch_label not in legend_handles:
            legend_handles[epoch_label] = plt.Line2D(
                [0], [0], color=epoch_color, lw=2, label=label_names.get(epoch_label, "unknown")
            )
        offset += n_samples

    plt.xlabel("Concatenated Sample Index")
    plt.ylabel(f"ICA Component {component + 1} Value")
    plt.title("Epochs in Original Time Series Order (Color Coded by Label)")
    plt.legend(handles=list(legend_handles.values()))
    plt.tight_layout()
    plt.show()

def plot_first3_and_concatenated_control_epochs(X_ica, orig_indices, y, control_data, window_size, control_only):
    """
    Plots the concatenated control epochs alongside the first 3 individual control epochs.
  
    Parameters:
        X_ica: Array of ICA components.
        orig_indices: Array of original epoch indices for each sample.
        y: Labels corresponding to each sample.
        control_data: Original control data (used to determine concatenation length).
        window_size: Window size used for plotting sliding windows.
        control_only: Boolean flag (if True then data contains only control epochs).
    """
    # Determine concatenated control signal.
    if not control_only:
        concat_length = control_data.shape[0]
        concatenated_control = X_ica[:concat_length, 0]
    else:
        concatenated_control = X_ica[:, 0]

    # Extract the first 3 control epochs (sorted order).
    control_epochs = []
    unique_epochs = np.unique(orig_indices)
    unique_epochs.sort()
    for epoch in unique_epochs:
        mask = orig_indices == epoch
        if y[mask][0] == 1:
            control_epochs.append(epoch)
            if len(control_epochs) == 3:
                break

    # Create a figure with 4 subplots: one for the concatenated and three for individual epochs.
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Plot concatenated control epochs.
    axs[0].plot(concatenated_control, color="blue")
    axs[0].set_title("Concatenated Control Epochs")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("IC1 Value")

    # Plot each of the first 3 individual control epochs.
    for i, epoch in enumerate(control_epochs):
        mask = orig_indices == epoch
        epoch_data = X_ica[mask, 0]
        axs[i+1].plot(epoch_data, color="green")
        axs[i+1].set_title(f"Control Epoch {i+1}")
        axs[i+1].set_xlabel("Sample Index")
        axs[i+1].set_ylabel("IC1 Value")

    plt.tight_layout()
    plt.show()


# ------------------ Main Flow ------------------
if __name__ == '__main__':
    window_size = 300
    step_size = 50
    control_only = False

    # Load data and concatenate epochs from control, left, and right conditions
    X, orig_indices, y, control_data, left_data, right_data, num_samples = load_data(subject=3, control_only=control_only)
    window_size = step_size = num_samples

    # Run ICA to obtain independent components
    X_ica = run_ica(X, n_components=5, standardized=False)
    
    plot_epochs_in_original_order(X_ica, orig_indices, y)    
    # Plot the independent components
    plot_ica_components(X_ica)

    plot_first3_and_concatenated_control_epochs(X_ica, orig_indices, y, control_data, window_size, control_only)

    # for _ in range(3):
    #     # Plot a single control epoch with sliding windows.
    #     unique_epochs = np.unique(orig_indices)
    #     np.random.shuffle(unique_epochs)
    #     control_epoch_data = None
    #     for epoch_id in unique_epochs:
    #         ep_mask = orig_indices == epoch_id
    #         if y[ep_mask][0] == 1:
    #             control_epoch_data = X_ica[ep_mask, 0]
    #             break
    #     if control_epoch_data is not None:
    #         plot_ica_epoch(control_epoch_data, window_size)

    
    # Compute sliding window features
    features, sw_labels = compute_sliding_windows(X_ica, y, window_size, step_size)
    
    # Cluster sliding window features
    gmm, cluster_labels = run_gmm(features, n_components=3)
    
    # Plot sliding window features and their GMM clusters
    plot_scatter_features(features, sw_labels)
    plot_gmm_clusters(features, cluster_labels)
    
    # Plot the concatenated control IC1 time series
    if not control_only:
        # Calculate unsorted boundaries: order: control, left, right
        boundary0 = 0
        boundary1 = control_data.shape[0]
        boundary2 = boundary1 + left_data.shape[0]
        boundary3 = boundary2 + right_data.shape[0]
        boundaries = [boundary0, boundary1, boundary2, boundary3]
        X_ica_unsorted = X_ica  # Preserved unsorted order
    
        # Plot unsorted time series with sliding window bands color-coded by GMM clusters
        block_colors = {1: "C0", 2: "C1", 3: "C2"}
        cluster_colors = {0: "pink", 1: "lightgreen", 2: "lightblue"}
        plot_unsorted_timeseries_with_clusters(X_ica_unsorted, boundaries, block_colors,
                                               window_size, step_size, sw_labels, cluster_labels, cluster_colors, ic=0, show_mean=False)
        # sil_score = silhouette_score(features, cluster_labels)
        # print(f"Silhouette Score: {sil_score:.3f}")
    else:
        # For control-only data, plot the control timeseries in black with cluster highlights.
        # X_ica contains only control epochs so no mask is needed.
        cluster_colors = {0: "pink", 1: "lightgreen", 2: "lightblue"}
        control_ic1 = X_ica[:, 0]
        plot_control_timeseries_with_clusters(control_ic1, window_size, step_size, cluster_labels, cluster_colors)
        # sil_score = silhouette_score(features, cluster_labels)
        # print(f"Silhouette Score: {sil_score:.3f}")
    
    