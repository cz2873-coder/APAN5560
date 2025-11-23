import torch
import matplotlib.pyplot as plt
from typing import Optional

from .trainer import DiffusionConfig


@torch.no_grad()
def p_sample_step(
    model: torch.nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    config: DiffusionConfig,
) -> torch.Tensor:
    """One reverse diffusion step p(x_{t-1} | x_t)."""
    betas = config.betas.to(x_t.device)
    alphas = config.alphas.to(x_t.device)
    alpha_bars = config.alpha_bars.to(x_t.device)

    beta_t = betas[t].view(-1, 1, 1, 1)
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)

    # Predict noise using the model
    eps_theta = model(x_t, t)

    # DDPM mean
    coeff1 = 1.0 / torch.sqrt(alpha_t)
    coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
    mean = coeff1 * (x_t - coeff2 * eps_theta)

    # Add noise except for t == 0
    noise = torch.randn_like(x_t)
    nonzero_mask = (t > 0).float().view(-1, 1, 1, 1)  # no noise when t == 0
    x_prev = mean + nonzero_mask * torch.sqrt(beta_t) * noise
    return x_prev


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    device: str,
    num_samples: int = 10,
    diffusion_steps: int = 1000,
    image_size: int = 28,
    plot: bool = False,
    seed: Optional[int] = None,
):
    """
    Generate images from a trained diffusion model.

    Returns:
        Tensor of shape (num_samples, 1, H, W)
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.to(device)
    model.eval()

    # Create schedules
    config = DiffusionConfig(num_timesteps=diffusion_steps)
    config.betas = config.betas.to(device)
    config.alphas = config.alphas.to(device)
    config.alpha_bars = config.alpha_bars.to(device)

    # Start from Gaussian noise
    x_t = torch.randn(num_samples, 1, image_size, image_size, device=device)

    for step in reversed(range(diffusion_steps)):
        t = torch.full((num_samples,), step, device=device, dtype=torch.long)
        x_t = p_sample_step(model, x_t, t, config)

    x_0 = x_t.clamp(-1.0, 1.0)

    if plot:
        # Plot on a grid (for your notebook / debugging)
        import math as _math

        num_cols = int(_math.ceil(_math.sqrt(num_samples)))
        num_rows = int(_math.ceil(num_samples / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        axes = axes.flatten()

        imgs = (x_0.cpu().squeeze(1) + 1.0) / 2.0  # map to [0, 1]
        for i in range(num_samples):
            ax = axes[i]
            ax.imshow(imgs[i], cmap="gray")
            ax.axis("off")

        for j in range(num_samples, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    return x_0
