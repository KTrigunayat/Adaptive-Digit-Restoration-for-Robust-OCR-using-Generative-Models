"""
train_vae.py
Training script for the Variational Autoencoder on clean MNIST images.

Usage:
    python -m src.models.train_vae
    python src/models/train_vae.py
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.vae import VAE
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    """Apply seed to torch, numpy, and python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: dict | None = None) -> None:
    """Train the VAE on clean MNIST and save weights to the configured checkpoint path."""
    if cfg is None:
        cfg = load_config()

    # Apply seed if present
    seed = cfg.get("seed") or cfg.get("data", {}).get("seed")
    if seed is not None:
        set_seed(int(seed))

    # Device selection
    device_name: str = cfg.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    # VAE hyperparameters
    vae_cfg = cfg["vae"]
    latent_dim: int = int(vae_cfg.get("latent_dim", 64))
    beta: float = float(vae_cfg.get("beta", 1.0))
    epochs: int = int(vae_cfg["epochs"])
    lr: float = float(vae_cfg.get("lr", 1e-4))
    checkpoint_path: str = vae_cfg.get("checkpoint", "checkpoints/vae.pth")

    # Create checkpoint directory if needed
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Data loading — clean MNIST, pixel values in [0, 1]
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=cfg.get("data", {}).get("raw_dir", "data/raw"),
        train=True,
        download=True,
        transform=transform,
    )
    batch_size: int = int(cfg.get("vae", {}).get("batch_size", 128))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device_name == "cuda"),
    )

    # Model and optimizer
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training VAE for {epochs} epochs on {device_name} | "
          f"latent_dim={latent_dim}, beta={beta}, lr={lr}")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss = VAE.loss(recon, images, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch [{epoch:3d}/{epochs}]  avg loss per sample: {avg_loss:.4f}")

    # Save weights
    torch.save(model.state_dict(), checkpoint_path)
    print(f"VAE weights saved to: {checkpoint_path}")


if __name__ == "__main__":
    train()
