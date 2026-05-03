"""
distortion_engine.py
Applies consistent synthetic corruptions to MNIST images for training and evaluation.
Supported: gaussian_noise, motion_blur, and spatial_masking.
"""

import numpy as np
import cv2
from skimage.util import random_noise


def add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    """Add Gaussian noise to an image with σ ~ Uniform(0.1, 0.5)."""
    sigma = np.random.uniform(0.1, 0.5)
    var = sigma ** 2
    noisy = random_noise(image, mode="gaussian", var=var)
    return (noisy * 255).astype(np.uint8)


def add_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply horizontal motion blur to an image using a 3×3 or 5×5 kernel."""
    if kernel_size not in (3, 5):
        kernel_size = 5
    # Build a horizontal motion kernel: a single row of ones, normalised
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(image, -1, kernel)


def add_masking(image: np.ndarray) -> np.ndarray:
    """Zero out exactly one contiguous 8×8 region at a random position."""
    corrupted = image.copy()
    h, w = image.shape[:2]
    # Ensure the 8×8 block fits within the image
    max_row = max(h - 8, 0)
    max_col = max(w - 8, 0)
    row = np.random.randint(0, max_row + 1)
    col = np.random.randint(0, max_col + 1)
    corrupted[row:row + 8, col:col + 8] = 0
    return corrupted


def apply_distortion(
    image: np.ndarray,
    distortion_type: str = "gaussian_noise",
    seed: int | None = None,
) -> np.ndarray:
    """
    Unified entry point for applying a distortion.

    Args:
        image: Input grayscale image as numpy array.
        distortion_type: One of 'gaussian_noise', 'motion_blur', 'spatial_masking'.
        seed: Optional integer seed for reproducible corruption generation.

    Returns:
        Corrupted image as numpy array with the same shape as the input.

    Raises:
        ValueError: If distortion_type is not one of the valid options.
    """
    dispatch = {
        "gaussian_noise": add_gaussian_noise,
        "motion_blur": add_blur,
        "spatial_masking": add_masking,
    }
    if distortion_type not in dispatch:
        raise ValueError(
            f"Unknown corruption type '{distortion_type}'. "
            f"Valid options: {list(dispatch.keys())}"
        )
    if seed is not None:
        np.random.seed(seed)
    return dispatch[distortion_type](image)
