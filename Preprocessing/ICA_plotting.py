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

def plot_unsorted_timeseries_with_clusters(X_ica_unsorted, boundaries, block_colors, window_size, step_size, labels, cluster_labels, cluster_colors):
    plt.figure(figsize=(12, 6))
    # Plot unsorted time series blocks
    plt.plot(np.arange(boundaries[0], boundaries[1]),
             X_ica_unsorted[boundaries[0]:boundaries[1], 0],
             color=block_colors[1], label="Control")
    plt.plot(np.arange(boundaries[1], boundaries[2]),
             X_ica_unsorted[boundaries[1]:boundaries[2], 0],
             color=block_colors[2], label="Tapping Left")
    plt.plot(np.arange(boundaries[2], boundaries[3]),
             X_ica_unsorted[boundaries[2]:boundaries[3], 0],
             color=block_colors[3], label="Tapping Right")
    plt.xlabel("Time (samples, unsorted epoch order: control, left, right)")
    plt.ylabel("ICA Component 1")
    plt.title("ICA Component 1 on Unsorted Epoch Data")

    # For each block/label add cluster overlay bands
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
            if win_counter < len(cl_cluster_labels):
                cl_lab = cl_cluster_labels[win_counter]
            else:
                cl_lab = 0
            win_counter += 1
            plt.axvspan(win_start, win_end, color=cluster_colors[cl_lab], alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()