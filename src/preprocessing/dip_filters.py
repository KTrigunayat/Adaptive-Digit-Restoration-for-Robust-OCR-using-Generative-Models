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


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    DIP preprocessing pipeline: median filter → histogram equalization.

    Applies a median filter (kernel_size=3) to suppress salt-and-pepper noise,
    then applies global histogram equalization to normalize pixel intensity contrast.

    Accepts uint8 or float32 grayscale arrays. If the input is float32, it is
    converted to uint8 before equalization and converted back to float32 on return.

    Args:
        image: Grayscale input array, shape (H, W), dtype uint8 or float32.

    Returns:
        Preprocessed NumPy array with the same spatial dimensions as the input.
        dtype matches the input dtype (uint8 → uint8, float32 → float32).
    """
    input_dtype = image.dtype

    # Step 1: median filter (works on both uint8 and float32 via cv2.medianBlur)
    filtered = median_filter(image, kernel_size=3)

    # Step 2: histogram equalization — cv2.equalizeHist requires uint8
    if input_dtype == np.float32:
        # Scale float32 [0.0, 1.0] → uint8 [0, 255]
        as_uint8 = (np.clip(filtered, 0.0, 1.0) * 255).astype(np.uint8)
        equalized = cv2.equalizeHist(as_uint8)
        # Convert back to float32 [0.0, 1.0]
        result = equalized.astype(np.float32) / 255.0
    else:
        # Ensure uint8 for equalizeHist
        as_uint8 = filtered.astype(np.uint8)
        result = cv2.equalizeHist(as_uint8)

    return result


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
