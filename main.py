"""
main.py
Pipeline entry point: corrupted MNIST input -> DIP preprocessing -> VAE encoding
-> Corruption classification -> Diffusion denoising -> VAE decoding -> OCR evaluation.
"""

import random
import torch
import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.vae import VAE
from src.models.unet import UNet
from src.models.ocr_classifier import OCRClassifier
from src.models.corruption_classifier import CorruptionClassifier
from src.core.diffusion_engine import DiffusionEngine
from src.preprocessing.dip_filters import preprocess
from data.distortion_engine import apply_distortion

log = get_logger("main")


def _load_checkpoint(model: torch.nn.Module, path: str, device: str) -> None:
    """Load state dict from path; raise FileNotFoundError if missing."""
    ckpt = Path(path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))


def _check_batch_size(tensor: torch.Tensor, expected_B: int, stage: str, prev_shape) -> None:
    """Raise RuntimeError if tensor's batch dim doesn't match expected_B."""
    if tensor.shape[0] != expected_B:
        raise RuntimeError(
            f"Batch size mismatch after stage '{stage}': "
            f"expected B={expected_B}, got shape {tuple(tensor.shape)} "
            f"(previous shape: {tuple(prev_shape)})"
        )


def run_pipeline(cfg: dict) -> dict:
    """
    Execute the full Adaptive Digit Restoration pipeline.

    Stages (in order):
        1. DIP preprocess
        2. VAE encode
        3. CorruptionClassifier predict_onehot
        4. DiffusionEngine reverse_process
        5. VAE decode

    Args:
        cfg: Configuration dict (from config.yaml via load_config).

    Returns:
        dict with keys: a_clean, a_corrupted, a_restored, mean_psnr, mean_elbo
    """
    # --- Device selection ---
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU.")
        device = "cpu"
    log.info(f"Running on device: {device}")

    # --- Load models and checkpoints ---
    vae = VAE(latent_dim=cfg["vae"]["latent_dim"]).to(device)
    _load_checkpoint(vae, cfg["vae"]["checkpoint"], device)
    vae.eval()

    corruption_clf = CorruptionClassifier().to(device)
    _load_checkpoint(corruption_clf, cfg["corruption_classifier"]["checkpoint"], device)
    corruption_clf.eval()

    unet = UNet().to(device)
    diffusion = DiffusionEngine(
        unet,
        timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        device=device,
    )
    _load_checkpoint(unet, cfg["diffusion"]["checkpoint"], device)
    unet.eval()

    ocr = OCRClassifier().to(device)
    _load_checkpoint(ocr, cfg["ocr"]["checkpoint"], device)
    ocr.eval()

    # --- Data loading ---
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root=cfg["data"]["raw_dir"], train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    clean_images, labels = next(iter(loader))
    clean_images = clean_images.to(device)
    labels = labels.to(device)
    B = clean_images.shape[0]

    # --- Apply distortion to produce corrupted batch ---
    distortion_type = cfg["data"]["distortion"]
    corrupted_np = np.stack([
        apply_distortion(img.squeeze().cpu().numpy(), distortion_type)
        for img in clean_images
    ])  # (B, 28, 28), uint8

    # --- Stage 1: DIP preprocess ---
    preprocessed_np = np.stack([preprocess(img) for img in corrupted_np])  # (B, 28, 28)
    # Convert to float32 tensor (B, 1, 28, 28) in [0, 1]
    if preprocessed_np.dtype == np.uint8:
        preprocessed = torch.tensor(preprocessed_np / 255.0, dtype=torch.float32)
    else:
        preprocessed = torch.tensor(preprocessed_np, dtype=torch.float32)
    preprocessed = preprocessed.unsqueeze(1).to(device)  # (B, 1, 28, 28)
    _check_batch_size(preprocessed, B, "DIP preprocess", clean_images.shape)
    log.info(f"Stage 'DIP preprocess' output shape: {tuple(preprocessed.shape)}")

    with torch.no_grad():
        # --- Stage 2: VAE encode ---
        mu, logvar = vae.encode(preprocessed)
        _check_batch_size(mu, B, "VAE encode", preprocessed.shape)
        log.info(f"Stage 'VAE encode' output shape (mu): {tuple(mu.shape)}")

        # --- Stage 3: CorruptionClassifier predict_onehot ---
        c = corruption_clf.predict_onehot(preprocessed)  # (B, 3)
        _check_batch_size(c, B, "CorruptionClassifier", preprocessed.shape)
        log.info(f"Stage 'CorruptionClassifier' output shape: {tuple(c.shape)}")

        # --- Stage 4: DiffusionEngine reverse_process ---
        # Start from the encoded latent (mu) as z_T approximation
        z_denoised = diffusion.reverse_process(mu, c)  # (B, 64)
        _check_batch_size(z_denoised, B, "DiffusionEngine reverse_process", mu.shape)
        log.info(f"Stage 'DiffusionEngine reverse_process' output shape: {tuple(z_denoised.shape)}")

        # --- Stage 5: VAE decode ---
        restored = vae.decode(z_denoised)  # (B, 1, 28, 28)
        _check_batch_size(restored, B, "VAE decode", z_denoised.shape)
        log.info(f"Stage 'VAE decode' output shape: {tuple(restored.shape)}")

    # --- Evaluation ---
    # OCR accuracy on clean images
    clean_preds = ocr.predict(clean_images)
    a_clean = (clean_preds == labels).float().mean().item()

    # OCR accuracy on corrupted images (convert corrupted_np to tensor)
    if corrupted_np.dtype == np.uint8:
        corrupted_tensor = torch.tensor(corrupted_np / 255.0, dtype=torch.float32)
    else:
        corrupted_tensor = torch.tensor(corrupted_np, dtype=torch.float32)
    corrupted_tensor = corrupted_tensor.unsqueeze(1).to(device)
    corrupted_preds = ocr.predict(corrupted_tensor)
    a_corrupted = (corrupted_preds == labels).float().mean().item()

    # OCR accuracy on restored images
    restored_preds = ocr.predict(restored)
    a_restored = (restored_preds == labels).float().mean().item()

    # PSNR between restored and clean
    mse = torch.mean((restored - clean_images) ** 2).item()
    mean_psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(1.0 / mse)

    # ELBO on clean images
    recon, mu_clean, logvar_clean = vae(clean_images)
    mean_elbo = -VAE.loss(recon, clean_images, mu_clean, logvar_clean).item() / B

    log.info(f"a_clean={a_clean:.4f}  a_corrupted={a_corrupted:.4f}  a_restored={a_restored:.4f}")
    log.info(f"mean_psnr={mean_psnr:.2f} dB  mean_elbo={mean_elbo:.4f}")

    return {
        "a_clean": a_clean,
        "a_corrupted": a_corrupted,
        "a_restored": a_restored,
        "mean_psnr": mean_psnr,
        "mean_elbo": mean_elbo,
    }


if __name__ == "__main__":
    cfg = load_config("config.yaml")

    # Apply global random seed for reproducibility (Requirement 9.4).
    _seed = cfg.get("seed", 42)
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

    results = run_pipeline(cfg)
    print(results)
