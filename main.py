"""
main.py
Pipeline entry point: corrupted MNIST input -> DIP preprocessing -> VAE encoding
-> Diffusion denoising -> OCR classification.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.models.vae import VAE
from src.models.unet import UNet
from src.models.ocr_classifier import OCRClassifier
from src.core.diffusion_engine import DiffusionEngine
from data.distortion_engine import apply_distortion

log = get_logger("main")


def run_pipeline(cfg: dict):
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU.")
        device = "cpu"

    log.info(f"Running on: {device}")

    # --- Data ---
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=cfg["data"]["raw_dir"], train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    images, labels = next(iter(loader))

    # --- Distortion ---
    corrupted = torch.stack([
        torch.tensor(
            apply_distortion(img.squeeze().numpy(), cfg["data"]["distortion"]) / 255.0,
            dtype=torch.float32,
        ).unsqueeze(0)
        for img in images
    ])
    log.info(f"Applied distortion: {cfg['data']['distortion']}")

    # --- VAE encode ---
    vae = VAE(latent_dim=cfg["vae"]["latent_dim"]).to(device)
    vae.eval()
    with torch.no_grad():
        recon, mu, logvar = vae(corrupted.to(device))
    log.info(f"VAE reconstruction shape: {recon.shape}")

    # --- Diffusion reverse ---
    unet = UNet().to(device)
    engine = DiffusionEngine(unet, timesteps=cfg["diffusion"]["timesteps"], device=device)
    restored = engine.reverse_process(recon)
    log.info(f"Diffusion restored shape: {restored.shape}")

    # --- OCR ---
    ocr = OCRClassifier().to(device)
    ocr.eval()
    preds = ocr.predict(restored)
    log.info(f"OCR predictions: {preds.tolist()}")
    log.info(f"Ground truth:    {labels.tolist()}")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    run_pipeline(cfg)
