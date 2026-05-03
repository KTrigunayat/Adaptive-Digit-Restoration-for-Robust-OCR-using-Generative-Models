"""
ocr_classifier.py
Standalone CNN-based OCR classifier for MNIST digit recognition.
Fully decoupled — can be benchmarked independently on raw, corrupted, or restored images.
"""

import torch
import torch.nn as nn


class OCRClassifier(nn.Module):
    """Simple CNN classifier. Input: (B, 1, 28, 28). Output: (B, 10) logits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
