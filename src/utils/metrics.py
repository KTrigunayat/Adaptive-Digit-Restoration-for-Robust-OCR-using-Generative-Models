"""
metrics.py
Pure functions for metric computation used in pipeline evaluation.

Provides:
  - compute_psnr: Peak Signal-to-Noise Ratio between restored and clean images
  - compute_elbo: Evidence Lower Bound on VAE reconstruction of images
  - compute_ocr_accuracy: Fraction of images correctly classified by OCR model
  - EvalReport: Dataclass summarising all evaluation metrics
"""

import math
from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import Tensor

from src.models.vae import VAE
from src.models.ocr_classifier import OCRClassifier


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def compute_psnr(restored: Tensor, clean: Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between restored and clean images.

    Both tensors must have the same shape and pixel values in [0, 1].

    Args:
        restored: Restored image tensor of shape (B, 1, H, W), values in [0, 1].
        clean:    Clean reference image tensor of same shape, values in [0, 1].

    Returns:
        PSNR value in dB. Returns float("inf") when the images are identical
        (MSE == 0), indicating perfect reconstruction.
    """
    if restored.shape != clean.shape:
        raise ValueError(
            f"Shape mismatch: restored {restored.shape} vs clean {clean.shape}"
        )

    with torch.no_grad():
        mse = torch.mean((restored.float() - clean.float()) ** 2).item()

    if mse == 0.0:
        return float("inf")

    # MAX_I = 1.0 for normalised [0, 1] images
    psnr = 10.0 * math.log10(1.0 / mse)
    return psnr


def compute_elbo(vae: VAE, images: Tensor) -> float:
    """Compute the mean ELBO (Evidence Lower Bound) for a batch of images.

    Runs a full VAE forward pass and computes the ELBO loss per image,
    then returns the mean over the batch.

    The ELBO is defined as:
        ELBO = -(ReconLoss + KL)
    where ReconLoss = BCE(recon, x, reduction='sum') and
          KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)).

    A higher (less negative) ELBO indicates better reconstruction quality.

    Args:
        vae:    A VAE model instance (in eval mode is recommended).
        images: Batch of images, shape (B, 1, H, W), values in [0, 1].

    Returns:
        Mean ELBO per image as a Python float.
    """
    with torch.no_grad():
        recon, mu, logvar = vae(images)
        # VAE.loss returns the total ELBO loss (sum over pixels and batch)
        total_loss = VAE.loss(recon, images, mu, logvar)

    batch_size = images.size(0)
    # Return mean ELBO per image (negated because VAE.loss is the negative ELBO)
    mean_elbo = -(total_loss.item() / batch_size)
    return mean_elbo


def compute_ocr_accuracy(
    ocr: OCRClassifier, images: Tensor, labels: Tensor
) -> float:
    """Compute the fraction of images correctly classified by the OCR model.

    Args:
        ocr:    An OCRClassifier model instance.
        images: Batch of images, shape (B, 1, H, W), values in [0, 1].
        labels: Ground-truth digit labels, shape (B,), integer class indices.

    Returns:
        Accuracy as a float in [0, 1] — the fraction of correctly classified
        images in the batch.
    """
    with torch.no_grad():
        predictions = ocr.predict(images)

    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluation report dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """Summary report produced after a full evaluation run.

    Attributes:
        a_clean:      OCR accuracy on clean (unmodified) images.
        a_corrupted:  OCR accuracy per corruption type,
                      e.g. {"gaussian_noise": 0.72, "motion_blur": 0.68, ...}.
        a_restored:   OCR accuracy on restored images per corruption type,
                      e.g. {"gaussian_noise": 0.91, "motion_blur": 0.88, ...}.
        mean_psnr:    Mean PSNR (dB) between restored and clean images per
                      corruption type,
                      e.g. {"gaussian_noise": 28.4, "motion_blur": 25.1, ...}.
        mean_elbo:    Mean VAE ELBO on clean images (scalar float).
    """

    a_clean: float
    a_corrupted: Dict[str, float] = field(default_factory=dict)
    a_restored: Dict[str, float] = field(default_factory=dict)
    mean_psnr: Dict[str, float] = field(default_factory=dict)
    mean_elbo: float = 0.0
