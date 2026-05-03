"""
distortion_engine.py
Applies consistent synthetic corruptions to MNIST images for training and evaluation.
Supported: Gaussian noise, blur, and random masking.
"""

import numpy as np
import cv2
from skimage.util import random_noise


def add_gaussian_noise(image: np.ndarray, var: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noisy = random_noise(image, mode="gaussian", var=var)
    return (noisy * 255).astype(np.uint8)


def add_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian blur to an image."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_masking(image: np.ndarray, mask_ratio: float = 0.3) -> np.ndarray:
    """Randomly zero out a fraction of pixels (masking corruption)."""
    corrupted = image.copy()
    mask = np.random.rand(*image.shape) < mask_ratio
    corrupted[mask] = 0
    return corrupted


def apply_distortion(image: np.ndarray, distortion_type: str = "noise", **kwargs) -> np.ndarray:
    """
    Unified entry point for applying a distortion.

    Args:
        image: Input grayscale image as numpy array.
        distortion_type: One of 'noise', 'blur', 'mask'.
        **kwargs: Additional parameters forwarded to the specific distortion function.

    Returns:
        Corrupted image as numpy array.
    """
    dispatch = {
        "noise": add_gaussian_noise,
        "blur": add_blur,
        "mask": add_masking,
    }
    if distortion_type not in dispatch:
        raise ValueError(f"Unknown distortion type '{distortion_type}'. Choose from {list(dispatch.keys())}.")
    return dispatch[distortion_type](image, **kwargs)
