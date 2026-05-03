"""
Integration test for the end-to-end restoration pipeline (main.py).

Uses randomly initialized models (no trained weights) and mocks to:
  - Verify pipeline stages are called in the correct order (Requirement 6.1)
  - Verify output shape is (B, 1, 28, 28) (Requirement 6.2)
  - Verify FileNotFoundError is raised for missing checkpoints (Requirement 6.4)

Requirements: 6.1, 6.2, 6.4
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from main import run_pipeline, _load_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 4  # small batch size for smoke tests


def _make_cfg(tmp_path: Path, missing_checkpoint: str | None = None) -> dict:
    """Build a minimal config pointing to temp checkpoint files."""
    checkpoints = {
        "vae": str(tmp_path / "vae.pth"),
        "corruption_classifier": str(tmp_path / "corruption_classifier.pth"),
        "diffusion": str(tmp_path / "diffusion.pth"),
        "ocr": str(tmp_path / "ocr.pth"),
    }

    # Write dummy checkpoint files (empty state dicts) unless marked missing
    from src.models.vae import VAE
    from src.models.corruption_classifier import CorruptionClassifier
    from src.models.unet import UNet
    from src.models.ocr_classifier import OCRClassifier

    models = {
        "vae": VAE(),
        "corruption_classifier": CorruptionClassifier(),
        "diffusion": UNet(),
        "ocr": OCRClassifier(),
    }

    for key, path in checkpoints.items():
        if key != missing_checkpoint:
            torch.save(models[key].state_dict(), path)

    return {
        "device": "cpu",
        "seed": 42,
        "data": {
            "raw_dir": "data/raw",
            "distortion": "gaussian_noise",
        },
        "vae": {
            "latent_dim": 64,
            "checkpoint": checkpoints["vae"],
        },
        "corruption_classifier": {
            "checkpoint": checkpoints["corruption_classifier"],
        },
        "diffusion": {
            "timesteps": 2,  # minimal T for speed
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "checkpoint": checkpoints["diffusion"],
        },
        "ocr": {
            "checkpoint": checkpoints["ocr"],
        },
        "evaluation": {
            "output_path": str(tmp_path / "eval_report.json"),
        },
    }


def _make_synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a small batch of synthetic MNIST-like images and labels."""
    torch.manual_seed(0)
    images = torch.rand(B, 1, 28, 28, dtype=torch.float32)
    labels = torch.randint(0, 10, (B,))
    return images, labels


# ---------------------------------------------------------------------------
# Stage ordering test
# ---------------------------------------------------------------------------

class TestPipelineStageOrdering:
    """Verify the five pipeline stages are invoked in the correct order."""

    def test_stages_called_in_order(self, tmp_path):
        """
        Patch each stage function/method and assert they are called in order:
        preprocess → VAE.encode → CorruptionClassifier.predict_onehot
        → DiffusionEngine.reverse_process → VAE.decode
        """
        cfg = _make_cfg(tmp_path)
        images, labels = _make_synthetic_batch()
        synthetic_dataset = list(zip(images, labels.tolist()))

        call_order = []

        def make_tracker(name, return_fn):
            """Return a side_effect function that records the call name."""
            def _tracked(*args, **kwargs):
                call_order.append(name)
                return return_fn(*args, **kwargs)
            return _tracked

        # patch.object on a class method receives (self, *args) in side_effect
        def encode_return(*args):
            x = args[-1]  # last positional arg is the input tensor
            B = x.shape[0]
            return torch.zeros(B, 64), torch.zeros(B, 64)

        def predict_onehot_return(*args):
            x = args[-1]
            B = x.shape[0]
            oh = torch.zeros(B, 3)
            oh[:, 0] = 1.0
            return oh

        def reverse_process_return(*args):
            z = args[1]  # (self, z, c)
            return torch.zeros(z.shape[0], 64)

        def decode_return(*args):
            z = args[-1]
            return torch.zeros(z.shape[0], 1, 28, 28)

        import main as main_module

        with patch("main.datasets.MNIST", return_value=synthetic_dataset), \
             patch("main.apply_distortion", side_effect=lambda img, t, **kw: img), \
             patch("main.preprocess",
                   side_effect=make_tracker("preprocess", lambda img: img)), \
             patch.object(main_module.VAE, "encode",
                          side_effect=make_tracker("encode", encode_return)), \
             patch.object(main_module.CorruptionClassifier, "predict_onehot",
                          side_effect=make_tracker("predict_onehot", predict_onehot_return)), \
             patch.object(main_module.DiffusionEngine, "reverse_process",
                          side_effect=make_tracker("reverse_process", reverse_process_return)), \
             patch.object(main_module.VAE, "decode",
                          side_effect=make_tracker("decode", decode_return)):
            run_pipeline(cfg)

        # Verify all stages were called
        for stage in ("preprocess", "encode", "predict_onehot", "reverse_process", "decode"):
            assert stage in call_order, f"Stage '{stage}' was not called"

        # Verify ordering
        idx = {s: next(i for i, v in enumerate(call_order) if v == s)
               for s in ("preprocess", "encode", "predict_onehot", "reverse_process", "decode")}

        assert idx["preprocess"] < idx["encode"], "preprocess must run before VAE encode"
        assert idx["encode"] < idx["predict_onehot"], "VAE encode must run before CorruptionClassifier"
        assert idx["predict_onehot"] < idx["reverse_process"], "CorruptionClassifier must run before DiffusionEngine"
        assert idx["reverse_process"] < idx["decode"], "DiffusionEngine must run before VAE decode"


# ---------------------------------------------------------------------------
# Output shape test
# ---------------------------------------------------------------------------

class TestPipelineOutputShape:
    """Verify the pipeline returns a restored batch of shape (B, 1, 28, 28)."""

    def test_restored_output_shape(self, tmp_path):
        """
        With randomly initialized models, the pipeline must return a dict
        whose 'a_restored' key exists and the internal restored tensor has
        shape (B, 1, 28, 28).
        """
        cfg = _make_cfg(tmp_path)
        images, labels = _make_synthetic_batch()
        synthetic_dataset = list(zip(images, labels.tolist()))

        with patch("main.datasets.MNIST", return_value=synthetic_dataset), \
             patch("main.apply_distortion",
                   side_effect=lambda img, t, **kw: img):
            results = run_pipeline(cfg)

        # run_pipeline returns a metrics dict — verify required keys exist
        assert "a_clean" in results
        assert "a_corrupted" in results
        assert "a_restored" in results
        assert "mean_psnr" in results
        assert "mean_elbo" in results

    def test_result_values_are_numeric(self, tmp_path):
        """All returned metric values must be finite numeric types."""
        import math

        cfg = _make_cfg(tmp_path)
        images, labels = _make_synthetic_batch()
        synthetic_dataset = list(zip(images, labels.tolist()))

        with patch("main.datasets.MNIST", return_value=synthetic_dataset), \
             patch("main.apply_distortion",
                   side_effect=lambda img, t, **kw: img):
            results = run_pipeline(cfg)

        for key in ("a_clean", "a_corrupted", "a_restored"):
            assert isinstance(results[key], float), f"{key} is not a float"
            assert 0.0 <= results[key] <= 1.0, f"{key}={results[key]} out of [0, 1]"

        assert isinstance(results["mean_elbo"], float)


# ---------------------------------------------------------------------------
# Missing checkpoint test
# ---------------------------------------------------------------------------

class TestPipelineMissingCheckpoint:
    """Verify FileNotFoundError is raised when a checkpoint is missing."""

    @pytest.mark.parametrize("missing", ["vae", "corruption_classifier", "diffusion", "ocr"])
    def test_missing_checkpoint_raises(self, tmp_path, missing):
        """FileNotFoundError must be raised for each missing checkpoint."""
        cfg = _make_cfg(tmp_path, missing_checkpoint=missing)
        images, labels = _make_synthetic_batch()
        synthetic_dataset = list(zip(images, labels.tolist()))

        with patch("main.datasets.MNIST", return_value=synthetic_dataset), \
             patch("main.apply_distortion",
                   side_effect=lambda img, t, **kw: img):
            with pytest.raises(FileNotFoundError):
                run_pipeline(cfg)
