import numpy as np
import matplotlib.pyplot as plt

# ------------------ Plotting Functions ------------------
def plot_scatter_features(features, labels, title="Scatter Plot of Sliding Window Features (IC1 vs IC2)"):
    colors = {1: "C0", 2: "C1", 3: "C2"}
    plt.figure(figsize=(8, 6))
    for cl in sorted(np.unique(labels)):
        cl_mask = labels == cl
        plt.scatter(features[cl_mask, 0], features[cl_mask, 1],
                    color=colors[cl], alpha=0.7, s=50, label=f"Class {cl}")
    plt.xlabel("Mean IC 1")
    plt.ylabel("Mean IC 2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_gmm_clusters(features, cluster_labels):
    plt.figure(figsize=(8, 6))
    for cl in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cl
        plt.scatter(features[cluster_mask, 0], features[cluster_mask, 1],
                    alpha=0.7, s=50, label=f"Cluster {cl}")
    plt.xlabel("Mean IC 1")
    plt.ylabel("Mean IC 2")
    plt.title("GMM Clustering on Sliding Window Features (IC1 vs IC2)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ica_epoch(control_epoch_data, window_size):
    plt.figure(figsize=(12, 4))
    plt.plot(control_epoch_data, label="IC1 Time Series", color="blue")
    n_samples = control_epoch_data.shape[0]
    for start in range(0, n_samples - window_size + 1, window_size):
        window_data = control_epoch_data[start:start+window_size]
        mean_val = np.mean(window_data)
        plt.axvline(x=start, color="red", linestyle="--", linewidth=1)
        plt.text(start, mean_val, f"{mean_val:.2f}", color="black",
                 fontsize=10, backgroundcolor="white", verticalalignment='bottom')
    plt.axvline(x=n_samples, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Sample Index")
    plt.ylabel("IC1 Value")
    plt.title("IC1 Time Series for a Control Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ica_components(X_ica):
    plt.figure(figsize=(10, 8))
    n_components = X_ica.shape[1]
    for i in range(n_components):
        plt.subplot(n_components, 1, i + 1)
        plt.plot(X_ica[:, i])
        plt.title(f'ICA Component {i + 1}')
    plt.tight_layout()
    plt.show()

def plot_concatenated_timeseries(control_ic1, window_size):
    plt.figure(figsize=(12, 4))
    plt.plot(control_ic1, label="Concatenated Control IC1", color="blue")
    n_samples = control_ic1.shape[0]
    for start in range(0, n_samples - window_size + 1, window_size):
        window_data = control_ic1[start:start+window_size]
        mean_val = np.mean(window_data)
        plt.axvline(x=start, color="red", linestyle="--", linewidth=1)
        plt.text(start, mean_val, f"{mean_val:.2f}", color="black",
                 fontsize=10, backgroundcolor="white", verticalalignment="bottom")
    plt.axvline(x=n_samples, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Sample Index")
    plt.ylabel("IC1 Value")
    plt.title("Concatenated Control IC1 Time Series with Sliding Windows")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_unsorted_timeseries_with_clusters(X_ica_unsorted, boundaries, block_colors, window_size, step_size, labels, cluster_labels, cluster_colors, ic=0, show_mean=True):

    plt.figure(figsize=(12, 6))
    # Plot unsorted time series blocks using the specified ICA component (ic)
    plt.plot(np.arange(boundaries[0], boundaries[1]),
             X_ica_unsorted[boundaries[0]:boundaries[1], ic],
             color=block_colors[1], label="Control")
    plt.plot(np.arange(boundaries[1], boundaries[2]),
             X_ica_unsorted[boundaries[1]:boundaries[2], ic],
             color=block_colors[2], label="Tapping Left")
    plt.plot(np.arange(boundaries[2], boundaries[3]),
             X_ica_unsorted[boundaries[2]:boundaries[3], ic],
             color=block_colors[3], label="Tapping Right")
    plt.xlabel("Time (samples, unsorted epoch order: control, left, right)")
    plt.ylabel(f"ICA Component {ic+1}")
    plt.title(f"ICA Component {ic+1} on Unsorted Epoch Data")
    
    # For each block/label add cluster overlay bands and optionally show window mean value.
    for cl in [1, 2, 3]:
        if cl == 1:
            offset, seg_end = boundaries[0], boundaries[1]
        elif cl == 2:
            offset, seg_end = boundaries[1], boundaries[2]
        elif cl == 3:
            offset, seg_end = boundaries[2], boundaries[3]
        segment_length = int(seg_end) - int(offset)
        cl_mask = (labels == cl)
        cl_cluster_labels = cluster_labels[cl_mask]
        win_counter = 0
        for start in range(0, segment_length - window_size + 1, step_size):
            win_start = offset + start
            win_end = win_start + window_size
            # Get current cluster label if available
            if win_counter < len(cl_cluster_labels):
                cl_lab = cl_cluster_labels[win_counter]
            else:
                cl_lab = 0
            # Highlight window with a colored overlay.
            plt.axvspan(win_start, win_end, color=cluster_colors[cl_lab], alpha=0.3)
            # Optionally compute the mean value for the window and mark it.
            if show_mean:
                window_data = X_ica_unsorted[win_start:win_end, ic]
                mean_val = np.mean(window_data)
                plt.axvline(x=win_start, color="red", linestyle="--", linewidth=1)
                plt.text(win_start, mean_val, f"{mean_val:.2f}", color="black",
                         fontsize=10, backgroundcolor="white", verticalalignment='bottom')
            win_counter += 1
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_unsorted_timeseries_with_clusters_two_classes(X_ica_unsorted, boundaries, block_colors, window_size, step_size, cluster_labels, cluster_colors, ic=0, show_mean=True):
    """
    Plot unsorted ICA time series with sliding window overlays for a two-class scenario.
    This function expects boundaries as a list of three indices:
        [start_of_control, end_of_control/start_of_tapping_left, end_of_tapping_left].
    Overlays sliding window bands colored by the provided cluster_labels.
    
    Parameters:
        X_ica_unsorted: 2D numpy array of ICA components where rows are time points.
        boundaries: List of three indices [b0, b1, b2] defining the two blocks.
        block_colors: Dictionary mapping block number (1 or 2) to a color.
        window_size: Number of samples per sliding window.
        step_size: Step size between windows.
        cluster_labels: Array of GMM cluster labels for each window over both blocks (concatenated).
        cluster_colors: Dictionary mapping each cluster label to a color.
        ic: ICA component index to plot.
        show_mean: If True, draw the mean value of each window.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    
    # There are two blocks: block 1 for control and block 2 for tapping left.
    for block in [1, 2]:
        if block == 1:
            offset, seg_end = boundaries[0], boundaries[1]
            label_name = "Control"
        else:
            offset, seg_end = boundaries[1], boundaries[2]
            label_name = "Tapping Left"
            
        # Plot the raw ICA time series for this block using the block color.
        plt.plot(range(offset, seg_end),
                 X_ica_unsorted[offset:seg_end, ic],
                 color=block_colors.get(block, "black"),
                 label=label_name)
        
        # Determine the segment length and number of sliding windows.
        segment_length = seg_end - offset
        # For each window, overlay color based on its cluster label.
        win_counter = 0
        for start in range(0, segment_length - window_size + 1, step_size):
            win_start = offset + start
            win_end = win_start + window_size
            # Get cluster label (global counter across both blocks)
            if win_counter < len(cluster_labels):
                cl_lab = cluster_labels[win_counter]
            else:
                cl_lab = 0
            plt.axvspan(win_start, win_end, color=cluster_colors.get(cl_lab, "gray"), alpha=0.3)
            if show_mean:
                window_data = X_ica_unsorted[win_start:win_end, ic]
                mean_val = window_data.mean()
                plt.axvline(x=win_start, color="red", linestyle="--", linewidth=1)
                plt.text(win_start, mean_val, f"{mean_val:.2f}", color="black",
                         fontsize=10, backgroundcolor="white", verticalalignment="bottom")
            win_counter += 1

    plt.xlabel("Time (samples, unsorted epoch order)")
    plt.ylabel(f"ICA Component {ic+1}")
    plt.title(f"ICA Component {ic+1} on Unsorted Data (2-Class Case)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_control_timeseries_with_clusters(control_ic1, window_size, step_size, cluster_labels, cluster_colors):
    """
    Plot the control IC1 time series with sliding window bands color-coded by GMM clusters.
    The line is plotted in black since only control epochs are used.
    
    Parameters:
        control_ic1: 1D numpy array of the concatenated IC1 values from control epochs.
        window_size: Window size used in the sliding window.
        step_size: Step size between sliding windows.
        cluster_labels: GMM cluster labels for each sliding window.
        cluster_colors: Dictionary mapping cluster labels to colors.
    """
    n_samples = control_ic1.shape[0]
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_samples), control_ic1, label="Control", color="black")

    # Highlight sliding window bands according to GMM clusters
    win_counter = 0
    for start in range(0, n_samples - window_size + 1, step_size):
        win_start = start
        win_end = start + window_size
        if win_counter < len(cluster_labels):
            cl_lab = cluster_labels[win_counter]
        else:
            cl_lab = 0
        plt.axvspan(win_start, win_end, color=cluster_colors.get(cl_lab, "gray"), alpha=0.3)
        win_counter += 1
    
    plt.xlabel("Sample Index")
    plt.ylabel("IC1 Value")
    plt.title("Control IC1 Time Series with GMM Cluster Highlighting")
    plt.legend()
    plt.tight_layout()
    plt.show()


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