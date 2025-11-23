import torch
from torch.utils.data import DataLoader
from typing import Callable


def get_beta_schedule(num_timesteps: int, start: float = 1e-4, end: float = 2e-2) -> torch.Tensor:
    """Linear beta schedule from start to end."""
    return torch.linspace(start, end, num_timesteps)


class DiffusionConfig:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_timesteps = num_timesteps
        self.betas = get_beta_schedule(num_timesteps, beta_start, beta_end)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    config: DiffusionConfig,
    noise: torch.Tensor | None = None,
):
    """Sample from q(x_t | x_0)."""
    if noise is None:
        noise = torch.randn_like(x0)

    alpha_bars = config.alpha_bars.to(x0.device)
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
    return x_t, noise


def train_diffusion(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epochs: int = 10,
    num_timesteps: int = 1000,
):
    """
    Train diffusion model to predict noise.

    Args:
        model: neural network that predicts noise epsilon_theta(x_t, t)
        data_loader: DataLoader yielding (x0, label) where x0 is image
        criterion: usually nn.MSELoss()
        optimizer: e.g. Adam
        device: 'cpu' or 'cuda'
        epochs: training epochs
        num_timesteps: total diffusion steps T
    """
    model.to(device)
    model.train()

    config = DiffusionConfig(num_timesteps=num_timesteps)
    alpha_bars = config.alpha_bars.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0

        for x0, _ in data_loader:
            x0 = x0.to(device)  # assume already normalized

            batch_size = x0.size(0)
            t = torch.randint(0, num_timesteps, (batch_size,), device=device, dtype=torch.long)

            # Forward diffusion: generate x_t from x_0
            noise = torch.randn_like(x0)
            a_bar = alpha_bars[t].view(-1, 1, 1, 1)
            x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

            optimizer.zero_grad()
            noise_pred = model(x_t, t)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        print(f"[Diffusion] Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    # 返回 model + config，方便后面生成样本
    return model, config
