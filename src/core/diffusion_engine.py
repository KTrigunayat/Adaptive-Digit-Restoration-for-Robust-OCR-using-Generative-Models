"""
diffusion_engine.py
Manages the forward (noise addition) and reverse (denoising) processes
for latent diffusion over MNIST digit representations.

The diffusion engine operates in the VAE latent space (64-dimensional vectors).
Latent vectors of shape (B, 64) are reshaped to (B, 1, 8, 8) for the UNet,
then reshaped back to (B, 64) after denoising.
"""

import torch
import torch.nn as nn
from src.models.unet import UNet


class DiffusionEngine:
    """
    DDPM-style diffusion engine operating in latent space.

    Args:
        unet: Trained U-Net noise predictor.
        timesteps: Total diffusion steps T.
        beta_start: Starting noise schedule value.
        beta_end: Ending noise schedule value.
        device: Torch device.
        vae: Optional VAE model used during training to encode images to latents.
             When provided, the VAE encoder is frozen (no gradient updates).
    """

    def __init__(
        self,
        unet: UNet,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
        vae=None,
    ):
        self.unet = unet.to(device)
        self.T = timesteps
        self.device = device
        self.vae = vae

        # Freeze VAE encoder if provided (requirement 5.5)
        if vae is not None:
            vae.to(device)
            vae.encoder.requires_grad_(False)

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, z0: torch.Tensor, t: torch.Tensor):
        """
        Add noise to z0 (latent vector) at timestep t (closed-form).

        Args:
            z0: Clean latent tensor of shape (B, 64).
            t: Timestep indices of shape (B,).

        Returns:
            z_t: Noisy latent of shape (B, 64).
            noise: The noise that was added, shape (B, 64).
        """
        noise = torch.randn_like(z0)
        # z0 is (B, 64) — use view(-1, 1) for broadcasting
        alpha_t = self.alpha_cumprod[t].view(-1, 1)
        z_t = torch.sqrt(alpha_t) * z0 + torch.sqrt(1 - alpha_t) * noise
        return z_t, noise

    @torch.no_grad()
    def reverse_process(self, z_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Iteratively denoise z_T back to z_0 using the U-Net, conditioned on c.

        Args:
            z_t: Noisy latent tensor at step T, shape (B, 64).
            c: Conditioning vector (1-hot corruption type), shape (B, 3).

        Returns:
            Denoised latent tensor z_0 of shape (B, 64).
        """
        z = z_t.to(self.device)
        c = c.to(self.device)
        B = z.shape[0]

        for t in reversed(range(self.T)):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Reshape latent (B, 64) -> (B, 1, 8, 8) for UNet
            z_spatial = z.view(B, 1, 8, 8)
            predicted_noise_spatial = self.unet(z_spatial, t_tensor, c)
            # Reshape predicted noise back to (B, 64)
            predicted_noise = predicted_noise_spatial.view(B, 64)

            alpha = self.alphas[t]
            alpha_hat = self.alpha_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)

            z = (1 / torch.sqrt(alpha)) * (
                z - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        return z

    def compute_loss(self, z0: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Sample a random timestep and compute MSE noise prediction loss.

        If a VAE is provided, z0 is treated as raw images and encoded first.
        Otherwise z0 is assumed to already be a latent vector of shape (B, 64).

        Args:
            z0: Either raw images (B, 1, 28, 28) if vae is set, or latent
                vectors (B, 64) if no vae is provided.
            c: Conditioning vector (1-hot corruption type), shape (B, 3).

        Returns:
            Scalar MSE loss tensor.
        """
        # Encode via frozen VAE encoder if available
        if self.vae is not None:
            with torch.no_grad():
                mu, logvar = self.vae.encode(z0)
                # Use the mean (mu) as the latent for training stability
                z0 = mu

        B = z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        z_t, noise = self.forward_process(z0, t)

        # Reshape latent (B, 64) -> (B, 1, 8, 8) for UNet
        z_t_spatial = z_t.view(B, 1, 8, 8)
        predicted_noise_spatial = self.unet(z_t_spatial, t, c)
        # Reshape predicted noise back to (B, 64)
        predicted_noise = predicted_noise_spatial.view(B, 64)

        return nn.functional.mse_loss(predicted_noise, noise)
