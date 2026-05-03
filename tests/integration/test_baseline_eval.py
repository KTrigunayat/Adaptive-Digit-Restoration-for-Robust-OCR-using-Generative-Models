"""
Integration test for experiments/baseline_ocr_eval.py

Runs run_baseline_eval() with a small synthetic dataset (no real MNIST download)
and asserts the output JSON has the correct structure.

Requirements: 8.4
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from experiments.baseline_ocr_eval import run_baseline_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 20  # small synthetic dataset size


def _make_synthetic_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images, labels) tensors mimicking a tiny MNIST test set."""
    torch.manual_seed(0)
    images = torch.rand(N, 1, 28, 28, dtype=torch.float32)
    labels = torch.randint(0, 10, (N,))
    return images, labels


def _make_cfg(output_path: str) -> dict:
    return {
        "seed": 42,
        "data": {"raw_dir": "data/raw", "distortion": "gaussian_noise"},
        "ocr": {"checkpoint": "checkpoints/ocr.pth"},
        "evaluation": {"output_path": output_path},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaselineEvalOutputStructure:
    """Verify run_baseline_eval produces a correctly structured JSON report."""

    def test_output_json_has_required_keys(self, tmp_path):
        """Output JSON must contain 'a_clean' and 'a_corrupted' keys."""
        output_file = str(tmp_path / "eval_report.json")
        cfg = _make_cfg(output_file)

        images, labels = _make_synthetic_mnist()

        # Patch MNIST dataset loading to return synthetic data
        with patch("experiments.baseline_ocr_eval.datasets.MNIST") as mock_mnist:
            mock_dataset = list(zip(images, labels.tolist()))
            mock_mnist.return_value = mock_dataset

            results = run_baseline_eval(cfg)

        assert "a_clean" in results, "Missing 'a_clean' key in results"
        assert "a_corrupted" in results, "Missing 'a_corrupted' key in results"

    def test_a_corrupted_has_all_three_corruption_types(self, tmp_path):
        """a_corrupted must contain entries for all three corruption types."""
        output_file = str(tmp_path / "eval_report.json")
        cfg = _make_cfg(output_file)

        images, labels = _make_synthetic_mnist()

        with patch("experiments.baseline_ocr_eval.datasets.MNIST") as mock_mnist:
            mock_dataset = list(zip(images, labels.tolist()))
            mock_mnist.return_value = mock_dataset

            results = run_baseline_eval(cfg)

        expected_types = {"gaussian_noise", "motion_blur", "spatial_masking"}
        assert set(results["a_corrupted"].keys()) == expected_types

    def test_accuracy_values_are_floats_in_unit_range(self, tmp_path):
        """All accuracy values must be floats in [0, 1]."""
        output_file = str(tmp_path / "eval_report.json")
        cfg = _make_cfg(output_file)

        images, labels = _make_synthetic_mnist()

        with patch("experiments.baseline_ocr_eval.datasets.MNIST") as mock_mnist:
            mock_dataset = list(zip(images, labels.tolist()))
            mock_mnist.return_value = mock_dataset

            results = run_baseline_eval(cfg)

        assert isinstance(results["a_clean"], float)
        assert 0.0 <= results["a_clean"] <= 1.0

        for corruption_type, acc in results["a_corrupted"].items():
            assert isinstance(acc, float), f"{corruption_type} accuracy is not a float"
            assert 0.0 <= acc <= 1.0, f"{corruption_type} accuracy {acc} out of [0, 1]"

    def test_output_json_is_written_to_configured_path(self, tmp_path):
        """Results must be written as valid JSON to the configured output path."""
        output_file = str(tmp_path / "subdir" / "eval_report.json")
        cfg = _make_cfg(output_file)

        images, labels = _make_synthetic_mnist()

        with patch("experiments.baseline_ocr_eval.datasets.MNIST") as mock_mnist:
            mock_dataset = list(zip(images, labels.tolist()))
            mock_mnist.return_value = mock_dataset

            run_baseline_eval(cfg)

        assert Path(output_file).exists(), "Output JSON file was not created"

        with open(output_file) as f:
            loaded = json.load(f)

        assert "a_clean" in loaded
        assert "a_corrupted" in loaded

    def test_output_json_values_match_returned_dict(self, tmp_path):
        """The written JSON must match the dict returned by run_baseline_eval."""
        output_file = str(tmp_path / "eval_report.json")
        cfg = _make_cfg(output_file)

        images, labels = _make_synthetic_mnist()

        with patch("experiments.baseline_ocr_eval.datasets.MNIST") as mock_mnist:
            mock_dataset = list(zip(images, labels.tolist()))
            mock_mnist.return_value = mock_dataset

            results = run_baseline_eval(cfg)

        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded["a_clean"] == pytest.approx(results["a_clean"])
        for k in results["a_corrupted"]:
            assert loaded["a_corrupted"][k] == pytest.approx(results["a_corrupted"][k])
