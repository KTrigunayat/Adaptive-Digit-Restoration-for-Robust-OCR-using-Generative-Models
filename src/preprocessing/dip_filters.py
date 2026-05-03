"""
dip_filters.py
Classical Digital Image Processing filters for pre-restoration cleanup.
Runs on CPU — decoupled from GPU-bound generative models.
"""

import cv2
import numpy as np


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian smoothing to reduce high-frequency noise."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Median filter — effective against salt-and-pepper noise."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def apply_filter(image: np.ndarray, filter_type: str = "gaussian", **kwargs) -> np.ndarray:
    """
    Unified filter dispatcher.

    Args:
        image: Grayscale input image.
        filter_type: 'gaussian' or 'median'.
        **kwargs: Forwarded to the selected filter.

    Returns:
        Filtered image.
    """
    dispatch = {
        "gaussian": gaussian_filter,
        "median": median_filter,
    }
    if filter_type not in dispatch:
        raise ValueError(f"Unknown filter '{filter_type}'. Choose from {list(dispatch.keys())}.")
    return dispatch[filter_type](image, **kwargs)
