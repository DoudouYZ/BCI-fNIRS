import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from Preprocessing import stack_epochs, get_epochs_for_subject

def visualize_ica_components(subject=0, window_length=1, nr_components=3):
    """
    Computes ICA for stacked epochs and visualizes the independent components.

    Args:
        subject (int): Subject number to analyze. Default is 0.
        window_length (float): Window length in seconds for epoch segmentation. Default is 1.
        nr_components (int): Number of independent components to extract and visualize. Default is 3.

    Returns:
        tuple: (fig, X_ica, y, ica_model) - The figure object, ICA-transformed data, labels, and the ICA object.
    """
    # Extract subject-specific epochs and stack them into features and labels.
    subject_epochs = get_epochs_for_subject(subject)
    X, y = stack_epochs(subject_epochs, s=window_length)

    # Compute ICA on the stacked epochs.
    ica_model = FastICA(n_components=nr_components, random_state=42, whiten="arbitrary-variance")
    X_ica = ica_model.fit_transform(X)

    # Create a grid of plots for the independent components.
    fig, axes = plt.subplots(nr_components, nr_components, figsize=(12, 12))
    components = [f'IC {i+1}' for i in range(nr_components)]

    for i in range(nr_components):
        for j in range(nr_components):
            ax = axes[i, j]
            if i == j:
                # Diagonal: Histogram of the component values.
                ax.hist(X_ica[:, i], bins=30, color='gray')
                ax.set_xlabel(components[i])
                ax.set_ylabel('Count')
            else:
                # Off-diagonals: Scatter plot of two components colored by the label.
                sc = ax.scatter(X_ica[:, j], X_ica[:, i], c=y, cmap='viridis', alpha=0.7)
                if i == nr_components - 1:
                    ax.set_xlabel(components[j])
                if j == 0:
                    ax.set_ylabel(components[i])

    plt.suptitle(f'Pairplot of the First {nr_components} Independent Components', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add a colorbar to the overall figure.
    if nr_components > 1:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label='Label')
    
    return fig, X_ica, y, ica_model

if __name__ == '__main__':
    fig, X_ica, y, ica_model = visualize_ica_components(subject=4, window_length=1, nr_components=3)
    
    plt.show()