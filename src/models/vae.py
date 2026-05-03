"""
vae.py
Variational Autoencoder for learning a compact latent representation of MNIST digits.
The encoder output feeds into the diffusion engine's latent space.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # (B, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # (B, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),                                # (B, 3136)
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # (B, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),   # (B, 1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, 64, 7, 7)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input image to (mu, logvar) each of shape (B, latent_dim)."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector z of shape (B, latent_dim) to (B, 1, 28, 28)."""
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass. Returns (recon, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss(
        recon_x: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        beta: float = 1.0,
    ) -> Tensor:
        """ELBO loss: BCE(recon, x, reduction='sum') + beta * KL.

        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        recon_loss = nn.functional.binary_cross_entropy(
            recon_x, x, reduction="sum"
        )
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld
