"""
visualize.py
Visualization utilities for comparing original, corrupted, and restored images.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_image_grid(images: list, titles: list = None, cols: int = 4, cmap: str = "gray"):
    """
    Display a grid of images.

    Args:
        images: List of numpy arrays or torch tensors (H, W) or (1, H, W).
        titles: Optional list of titles per image.
        cols: Number of columns in the grid.
        cmap: Colormap.
    """
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).flatten()

    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.squeeze().cpu().numpy()
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis("off")
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses: list, title: str = "Training Loss"):
    """Plot a simple loss curve."""
    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
