"""
train_diffusion.py
Training script for the Conditional Latent Diffusion Model (DiffusionEngine).

Trains the UNet noise predictor on synthetically corrupted MNIST latents,
conditioned on the ground-truth corruption type label (1-hot encoded).
The VAE encoder is loaded from a pre-trained checkpoint and kept frozen
throughout training.

Usage:
    python -m src.core.train_diffusion
    python src/core/train_diffusion.py
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from data.distortion_engine import apply_distortion
from src.core.diffusion_engine import DiffusionEngine
from src.models.corruption_classifier import CORRUPTION_TYPES
from src.models.unet import UNet
from src.models.vae import VAE
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

class CorruptedMNISTLatentDataset(Dataset):
    """
    Wraps a clean MNIST dataset and applies a random corruption type to each
    image on-the-fly.  Returns the corrupted image tensor and a 1-hot
    conditioning vector representing the corruption type.

    The VAE encoding is done in the training loop (not here) so that the
    dataset remains lightweight and the VAE can be on GPU.

    Returns:
        (corrupted_image_tensor, onehot_label) where:
            - corrupted_image_tensor: (1, 28, 28) float32 in [0, 1]
            - onehot_label: (3,) float32 1-hot vector
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

        # Build 1-hot conditioning vector
        onehot = torch.zeros(len(CORRUPTION_TYPES), dtype=torch.float32)
        onehot[corruption_idx] = 1.0

        return corrupted_tensor, onehot


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict | None = None) -> None:
    """
    Train the DiffusionEngine UNet and save weights to the configured path.

    Steps:
    1. Load frozen VAE from checkpoints/vae.pth
    2. Build DiffusionEngine with the frozen VAE
    3. Train UNet on corrupted MNIST latents conditioned on corruption type
    4. Save UNet weights to checkpoints/diffusion.pth
    """
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
    diff_cfg = cfg.get("diffusion", {})
    timesteps: int = int(diff_cfg.get("timesteps", 1000))
    beta_start: float = float(diff_cfg.get("beta_start", 1e-4))
    beta_end: float = float(diff_cfg.get("beta_end", 0.02))
    epochs: int = int(diff_cfg.get("epochs", 30))
    lr: float = float(diff_cfg.get("lr", 1e-4))
    checkpoint_path: str = diff_cfg.get("checkpoint", "checkpoints/diffusion.pth")
    batch_size: int = int(diff_cfg.get("batch_size", 128))

    vae_cfg = cfg.get("vae", {})
    latent_dim: int = int(vae_cfg.get("latent_dim", 64))
    vae_checkpoint: str = vae_cfg.get("checkpoint", "checkpoints/vae.pth")

    # Create checkpoint directory if needed
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Load frozen VAE
    # ---------------------------------------------------------------------------
    if not Path(vae_checkpoint).exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found: {vae_checkpoint}. "
            "Please train the VAE first using src/models/train_vae.py"
        )

    vae = VAE(latent_dim=latent_dim)
    vae.load_state_dict(torch.load(vae_checkpoint, map_location=device))
    vae.to(device)
    vae.eval()
    # Freeze the encoder (requirement 5.5)
    vae.encoder.requires_grad_(False)
    print(f"Loaded frozen VAE from: {vae_checkpoint}")

    # ---------------------------------------------------------------------------
    # Build DiffusionEngine with UNet
    # ---------------------------------------------------------------------------
    # UNet operates on (B, 1, 8, 8) spatial tensors (reshaped from (B, 64) latents)
    unet = UNet(in_channels=1, base_ch=32, t_dim=64, cond_dim=3)

    engine = DiffusionEngine(
        unet=unet,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=str(device),
        vae=vae,
    )

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
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

    data_seed = int(seed) if seed is not None else None
    train_dataset = CorruptedMNISTLatentDataset(mnist_train, seed=data_seed)
    val_dataset = CorruptedMNISTLatentDataset(mnist_val, seed=data_seed)

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

    # ---------------------------------------------------------------------------
    # Optimizer — only UNet parameters (VAE encoder is frozen)
    # ---------------------------------------------------------------------------
    optimizer = torch.optim.Adam(engine.unet.parameters(), lr=lr)

    print(
        f"Training DiffusionEngine for {epochs} epochs on {device_name} | "
        f"timesteps={timesteps}, beta_start={beta_start}, beta_end={beta_end}, "
        f"lr={lr}, batch_size={batch_size}"
    )

    for epoch in range(1, epochs + 1):
        # --- Training ---
        engine.unet.train()
        train_loss = 0.0
        train_batches = 0

        for images, onehot in train_loader:
            images = images.to(device)
            onehot = onehot.to(device)

            optimizer.zero_grad()
            # compute_loss encodes images via frozen VAE, then computes MSE loss
            loss = engine.compute_loss(images, onehot)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # --- Validation ---
        engine.unet.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, onehot in val_loader:
                images = images.to(device)
                onehot = onehot.to(device)
                loss = engine.compute_loss(images, onehot)
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"train_loss: {avg_train_loss:.6f}  "
            f"val_loss: {avg_val_loss:.6f}"
        )

    # Save UNet weights only (not the full engine)
    torch.save(engine.unet.state_dict(), checkpoint_path)
    print(f"DiffusionEngine UNet weights saved to: {checkpoint_path}")


if __name__ == "__main__":
    train()
