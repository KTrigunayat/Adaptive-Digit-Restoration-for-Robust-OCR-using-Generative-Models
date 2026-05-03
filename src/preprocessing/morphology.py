"""
morphology.py
Morphological operations for binary digit cleanup after DIP filtering.
"""

import cv2
import numpy as np


def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Erode to remove small noise blobs."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(image, kernel, iterations=iterations)


def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Dilate to restore digit stroke thickness after erosion."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(image, kernel, iterations=iterations)


def opening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological opening: erosion followed by dilation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological closing: dilation followed by erosion — fills small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
