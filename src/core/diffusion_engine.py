"""
diffusion_engine.py
Manages the forward (noise addition) and reverse (denoising) processes
for latent diffusion over MNIST digit representations.
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
    """

    def __init__(
        self,
        unet: UNet,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ):
        self.unet = unet.to(device)
        self.T = timesteps
        self.device = device

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Add noise to x0 at timestep t (closed-form).

        Returns:
            x_t: Noisy sample.
            noise: The noise that was added.
        """
        noise = torch.randn_like(x0)
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise

    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Iteratively denoise x_T back to x_0 using the U-Net.

        Args:
            x_t: Noisy latent tensor at step T.

        Returns:
            Denoised tensor x_0.
        """
        x = x_t.to(self.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.unet(x, t_tensor)

            alpha = self.alphas[t]
            alpha_hat = self.alpha_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        return x

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """Sample a random timestep and compute MSE noise prediction loss."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        x_t, noise = self.forward_process(x0, t)
        predicted_noise = self.unet(x_t, t)
        return nn.functional.mse_loss(predicted_noise, noise)
