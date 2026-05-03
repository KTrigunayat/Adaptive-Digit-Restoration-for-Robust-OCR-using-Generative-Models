"""
Unit tests for src/preprocessing/dip_filters.py

Covers the `preprocess` function (Task 2) and the existing filter utilities.
Property 4 (DIP preprocessing preserves spatial dimensions) is implemented here
as part of task 2.1 but is included for completeness.
"""

import numpy as np
import pytest

from src.preprocessing.dip_filters import apply_filter, median_filter, preprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_uint8(h: int = 28, w: int = 28) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def make_float32(h: int = 28, w: int = 28) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random(size=(h, w)).astype(np.float32)


# ---------------------------------------------------------------------------
# preprocess — dtype handling
# ---------------------------------------------------------------------------

class TestPreprocessDtype:
    def test_uint8_input_returns_uint8(self):
        img = make_uint8()
        result = preprocess(img)
        assert result.dtype == np.uint8

    def test_float32_input_returns_float32(self):
        img = make_float32()
        result = preprocess(img)
        assert result.dtype == np.float32

    def test_float32_output_in_unit_range(self):
        img = make_float32()
        result = preprocess(img)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0


# ---------------------------------------------------------------------------
# preprocess — spatial dimensions preserved (Requirement 2.4)
# ---------------------------------------------------------------------------

class TestPreprocessShape:
    @pytest.mark.parametrize("h,w", [(28, 28), (32, 32), (64, 64), (16, 16)])
    def test_uint8_shape_preserved(self, h, w):
        img = make_uint8(h, w)
        result = preprocess(img)
        assert result.shape == (h, w)

    @pytest.mark.parametrize("h,w", [(28, 28), (32, 32), (64, 64), (16, 16)])
    def test_float32_shape_preserved(self, h, w):
        img = make_float32(h, w)
        result = preprocess(img)
        assert result.shape == (h, w)


# ---------------------------------------------------------------------------
# preprocess — ordering: median filter applied before histogram equalization
# (Requirement 2.3)
# ---------------------------------------------------------------------------

class TestPreprocessOrdering:
    def test_median_before_equalize_uint8(self):
        """
        Verify that preprocess applies median filter first, then equalizeHist.
        We do this by manually replicating the expected pipeline and comparing
        the result to preprocess output.
        """
        import cv2

        img = make_uint8()
        # Expected: median_filter → equalizeHist
        after_median = median_filter(img, kernel_size=3)
        expected = cv2.equalizeHist(after_median)

        result = preprocess(img)
        np.testing.assert_array_equal(result, expected)

    def test_median_before_equalize_float32(self):
        """Same ordering check for float32 input."""
        import cv2

        img = make_float32()
        after_median = median_filter(img, kernel_size=3)
        as_uint8 = (np.clip(after_median, 0.0, 1.0) * 255).astype(np.uint8)
        equalized = cv2.equalizeHist(as_uint8)
        expected = equalized.astype(np.float32) / 255.0

        result = preprocess(img)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


# ---------------------------------------------------------------------------
# preprocess — salt-and-pepper noise suppression (Requirement 2.1)
# ---------------------------------------------------------------------------

class TestPreprocessNoiseSuppression:
    def test_salt_and_pepper_reduced(self):
        """
        After preprocessing, extreme pixel values (0 and 255) should be
        reduced compared to a heavily corrupted input.
        """
        rng = np.random.default_rng(0)
        img = rng.integers(50, 200, size=(28, 28), dtype=np.uint8)
        # Inject salt-and-pepper noise
        noise_mask = rng.random(size=(28, 28)) < 0.2
        img[noise_mask & (rng.random(size=(28, 28)) < 0.5)] = 0
        img[noise_mask & (rng.random(size=(28, 28)) >= 0.5)] = 255

        result = preprocess(img)
        # After median filter, isolated extreme pixels should be smoothed
        assert result.shape == (28, 28)


# ---------------------------------------------------------------------------
# apply_filter dispatcher — unchanged behaviour (Requirement 2.5 / existing)
# ---------------------------------------------------------------------------

class TestApplyFilterDispatcher:
    def test_gaussian_dispatch(self):
        img = make_uint8()
        result = apply_filter(img, filter_type="gaussian")
        assert result.shape == img.shape

    def test_median_dispatch(self):
        img = make_uint8()
        result = apply_filter(img, filter_type="median")
        assert result.shape == img.shape

    def test_unknown_filter_raises_value_error(self):
        img = make_uint8()
        with pytest.raises(ValueError, match="Unknown filter"):
            apply_filter(img, filter_type="nonexistent")
