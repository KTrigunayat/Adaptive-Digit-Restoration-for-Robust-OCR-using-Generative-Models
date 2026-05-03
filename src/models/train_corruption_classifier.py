"""
train_corruption_classifier.py
Training script for the CorruptionClassifier on synthetically corrupted MNIST images.

Generates training data on-the-fly by applying each of the three corruption types
(gaussian_noise, motion_blur, spatial_masking) to clean MNIST images using
apply_distortion from data/distortion_engine.py.

Usage:
    python -m src.models.train_corruption_classifier
    python src/models/train_corruption_classifier.py
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from data.distortion_engine import apply_distortion
from src.models.corruption_classifier import CorruptionClassifier, CORRUPTION_TYPES
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Apply seed to torch, numpy, and python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CorruptedMNISTDataset(Dataset):
    """
    Wraps a clean MNIST dataset and applies a random corruption type to each
    image on-the-fly.  Each sample is assigned one of the three corruption
    types in round-robin order so that the dataset is balanced.

    Returns:
        (image_tensor, label) where label ∈ {0, 1, 2} corresponds to
        [gaussian_noise, motion_blur, spatial_masking].
    """

    def __init__(self, mnist_dataset: Dataset, seed: int | None = None) -> None:
        self.mnist = mnist_dataset
        self.seed = seed
        # Pre-assign corruption type labels in round-robin order for balance
        n = len(mnist_dataset)
        self.corruption_labels = [i % len(CORRUPTION_TYPES) for i in range(n)]

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int):
        image_tensor, _ = self.mnist[idx]  # (1, 28, 28) float32 in [0, 1]

        corruption_idx = self.corruption_labels[idx]
        corruption_type = CORRUPTION_TYPES[corruption_idx]

        # Convert to uint8 numpy array for apply_distortion
        image_np = (image_tensor.squeeze(0).numpy() * 255).astype(np.uint8)

        # Use a deterministic per-sample seed derived from the global seed
        sample_seed = None
        if self.seed is not None:
            sample_seed = (self.seed + idx) % (2 ** 31)

        corrupted_np = apply_distortion(image_np, corruption_type, seed=sample_seed)

        # Convert back to float32 tensor in [0, 1]
        corrupted_tensor = torch.from_numpy(
            corrupted_np.astype(np.float32) / 255.0
        ).unsqueeze(0)  # (1, 28, 28)

        return corrupted_tensor, corruption_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict | None = None) -> None:
    """Train the CorruptionClassifier and save weights to the configured path."""
    if cfg is None:
        cfg = load_config()

    # Seed
    seed = cfg.get("seed") or cfg.get("data", {}).get("seed")
    if seed is not None:
        set_seed(int(seed))

    # Device
    device_name: str = cfg.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    # Hyperparameters from config
    cc_cfg = cfg.get("corruption_classifier", {})
    epochs: int = int(cc_cfg.get("epochs", 15))
    lr: float = float(cc_cfg.get("lr", 1e-4))
    checkpoint_path: str = cc_cfg.get("checkpoint", "checkpoints/corruption_classifier.pth")
    batch_size: int = int(cc_cfg.get("batch_size", 128))

    # Create checkpoint directory if needed
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Load clean MNIST
    raw_dir: str = cfg.get("data", {}).get("raw_dir", "data/raw")
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(
        root=raw_dir,
        train=True,
        download=True,
        transform=transform,
    )
    mnist_val = datasets.MNIST(
        root=raw_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Wrap with corruption augmentation
    data_seed = int(seed) if seed is not None else None
    train_dataset = CorruptedMNISTDataset(mnist_train, seed=data_seed)
    val_dataset = CorruptedMNISTDataset(mnist_val, seed=data_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device_name == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device_name == "cuda"),
    )

    # Model, loss, optimizer
    model = CorruptionClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"Training CorruptionClassifier for {epochs} epochs on {device_name} | "
        f"lr={lr}, batch_size={batch_size}"
    )

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images: Tensor = images.to(device)
            labels: Tensor = labels.to(device)

            optimizer.zero_grad()
            # forward returns softmax probs; CrossEntropyLoss expects logits,
            # so we pass through the network's internal logits instead.
            # Re-compute logits directly to avoid double-softmax.
            logits = _forward_logits(model, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = _forward_logits(model, images)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"loss: {avg_train_loss:.4f}  "
            f"train_acc: {train_acc:.4f}  "
            f"val_acc: {val_acc:.4f}"
        )

    # Save weights
    torch.save(model.state_dict(), checkpoint_path)
    print(f"CorruptionClassifier weights saved to: {checkpoint_path}")


def _forward_logits(model: CorruptionClassifier, x: Tensor) -> Tensor:
    """
    Run the model's feature extractor + classifier and return raw logits
    (before softmax).  This avoids the double-softmax problem when using
    CrossEntropyLoss, which applies log-softmax internally.
    """
    features = model.features(x)
    # classifier: Flatten → Linear → ReLU → Dropout → Linear
    # We need the raw output of the final Linear, not the softmax from forward()
    flat = features.flatten(start_dim=1)
    h = model.classifier[1](flat)   # Linear(3136→128)
    h = model.classifier[2](h)      # ReLU
    h = model.classifier[3](h)      # Dropout
    logits = model.classifier[4](h) # Linear(128→3)
    return logits


if __name__ == "__main__":
    train()
