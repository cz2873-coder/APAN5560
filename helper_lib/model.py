import torch
import torch.nn as nn
import math


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: Long tensor of shape (N,)
        dim: embedding dimension
    Returns:
        Tensor of shape (N, dim)
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class DiffusionUNet(nn.Module):
    """A very small UNet-like network for diffusion on 1xHxW images."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU(),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(4, base_channels * 2),
            nn.ReLU(),
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.ReLU(),
        )
        self.dec2 = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise given noised image x_t and timestep t.

        Args:
            x: Tensor of shape (N, C, H, W)
            t: Long tensor of shape (N,)
        """
        # Time embedding
        temb = sinusoidal_time_embedding(t, self.time_dim)
        temb = self.time_mlp(temb)  # (N, time_dim)
        temb = temb[:, :, None, None]  # (N, time_dim, 1, 1)

        # We only use first x.size(1) channels of time embedding as bias
        h1 = self.enc1(x + temb[:, : x.size(1)])
        h2 = self.enc2(h1)

        h = self.bottleneck(h2)

        h = self.dec1(h)
        h = self.dec2(h)
        return h


def get_model(model_name: str):
    """
    Return the requested model.

    Currently supports:
        - 'Diffusion'
    """
    name = model_name.lower()
    if name == "diffusion":
        return DiffusionUNet()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
