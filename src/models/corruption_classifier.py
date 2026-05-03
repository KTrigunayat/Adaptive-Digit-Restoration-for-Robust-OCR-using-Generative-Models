"""
corruption_classifier.py
Lightweight CNN that predicts the corruption type from a preprocessed MNIST image.

Supported corruption types (class indices):
    0 — gaussian_noise
    1 — motion_blur
    2 — spatial_masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Canonical ordering of corruption types (index → name)
CORRUPTION_TYPES = ["gaussian_noise", "motion_blur", "spatial_masking"]


class CorruptionClassifier(nn.Module):
    """
    CNN classifier for corruption type prediction.

    Architecture:
        Input: (B, 1, 28, 28)
        Conv2d(1→32, k=3, pad=1) → ReLU → MaxPool2d(2)  → (B, 32, 14, 14)
        Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool2d(2) → (B, 64, 7, 7)
        Flatten → Linear(3136→128) → ReLU → Dropout(0.3)
        Linear(128→3) → softmax probabilities
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 64, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # (B, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Softmax probability tensor of shape (B, 3).
        """
        logits = self.classifier(self.features(x))
        return F.softmax(logits, dim=1)

    def predict_onehot(self, x: Tensor) -> Tensor:
        """
        Predict corruption type as a 1-hot encoded vector.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            1-hot tensor of shape (B, 3) where each row has exactly one 1.
        """
        with torch.no_grad():
            probs = self.forward(x)                          # (B, 3)
            indices = torch.argmax(probs, dim=1, keepdim=True)  # (B, 1)
            onehot = torch.zeros_like(probs)
            onehot.scatter_(1, indices, 1.0)
            return onehot
