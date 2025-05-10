import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

import numpy as np
from sklearn.decomposition import PCA
from Preprocessing import stack_epochs, get_epochs_for_subject
import argparse

import matplotlib.pyplot as plt
def visualize_pca_components(subject=0, window_length=1, nr_pcs=3):
    """
    Computes PCA for stacked epochs and visualizes the principal components.
    
    Args:
        subject (int): Subject number to analyze. Default is 0.
        window_length (float): Window length in seconds for epoch segmentation. Default is 1.
        nr_pcs (int): Number of principal components to extract and visualize. Default is 3.
    
    Returns:
        tuple: (fig, X_pca, y, pca) - The figure object, PCA-transformed data, labels, and PCA object.
    """
    subject_epochs = get_epochs_for_subject(subject)
    # Compute features and labels from the stacked epochs
    X, y = stack_epochs(subject_epochs, s=window_length)

    # Perform PCA reducing to specified number of components
    pca = PCA(n_components=nr_pcs)
    X_pca = pca.fit_transform(X)

    # Create a grid of plots for the principal components
    fig, axes = plt.subplots(nr_pcs, nr_pcs, figsize=(12, 12))
    pcs = [f'PC {i+1}' for i in range(nr_pcs)]

    for i in range(nr_pcs):
        for j in range(nr_pcs):
            ax = axes[i, j]
            if i == j:
                # Plot a histogram on the diagonal
                ax.hist(X_pca[:, i], bins=30, color='gray')
                ax.set_xlabel(pcs[i])
                ax.set_ylabel('Count')
            else:
                # Scatter plot for the off-diagonals
                sc = ax.scatter(X_pca[:, j], X_pca[:, i], c=y, cmap='viridis', alpha=0.7)
                if i == nr_pcs-1:
                    ax.set_xlabel(pcs[j])
                if j == 0:
                    ax.set_ylabel(pcs[i])

    plt.suptitle(f'Pairplot of the First {nr_pcs} Principal Components', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Only add colorbar if there are off-diagonal plots
    if nr_pcs > 1:
        fig.colorbar(sc, ax=axes.ravel().tolist(), label='Label')
    
    return fig, X_pca, y, pca
if __name__ == '__main__':
    # 0.128 for full sample rate
    fig, X_pca, y, pca = visualize_pca_components(subject=2, window_length=0.128, nr_pcs=3)
    print("it should show now ...")
    plt.show()
