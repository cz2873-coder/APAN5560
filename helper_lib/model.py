# helper_lib/model.py
import torch
import torch.nn as nn


# -----------------------------------------------------------
# Generator
# -----------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, 1, 28, 28)


# -----------------------------------------------------------
# Discriminator
# -----------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


# -----------------------------------------------------------
# GAN container
# -----------------------------------------------------------
class GAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()


# -----------------------------------------------------------
# get_model() — required by class activity
# -----------------------------------------------------------
def get_model(model_name: str):
    name = model_name.lower()

    if name == "gan":
        return GAN(latent_dim=100)

    raise ValueError(f"Unknown model name: {model_name}")
