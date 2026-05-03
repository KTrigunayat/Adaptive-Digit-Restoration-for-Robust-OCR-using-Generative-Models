"""
unet.py
U-Net noise prediction network used as the backbone for the diffusion denoising process.
Accepts a noisy latent + timestep embedding and predicts the noise to subtract.
"""

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        # t: (B,) -> (B, dim)
        return self.net(t.float().unsqueeze(-1))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
        self.t_proj = nn.Linear(t_dim, out_ch)

    def forward(self, x, t_emb):
        h = self.conv(x)
        h = h + self.t_proj(t_emb)[:, :, None, None]
        return h


class UNet(nn.Module):
    """Lightweight U-Net for 28x28 single-channel latent denoising."""

    def __init__(self, in_channels: int = 1, base_ch: int = 32, t_dim: int = 64):
        super().__init__()
        self.t_emb = TimeEmbedding(t_dim)

        self.enc1 = ConvBlock(in_channels, base_ch, t_dim)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, t_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4, t_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2, t_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 2, base_ch, t_dim)

        self.out = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.t_emb(t)

        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        b = self.bottleneck(self.pool(e2), t_emb)

        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1), t_emb)

        return self.out(d2)
