# helper_lib/trainer.py
import torch
from torch import nn, optim


def train_gan(model, data_loader, device="cpu", epochs=3, lr_g=2e-4, lr_d=2e-4):
    device = torch.device(device)

    G = model.generator.to(device)
    D = model.discriminator.to(device)
    latent_dim = model.latent_dim

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    for epoch in range(1, epochs + 1):
        for real, _ in data_loader:
            real = real.to(device)
            batch_size = real.size(0)

            valid = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)

            # -------------------------
            # Train Discriminator
            # -------------------------
            z = torch.randn(batch_size, latent_dim, device=device)
            fake = G(z)

            D_real = D(real)
            D_fake = D(fake.detach())

            loss_D = criterion(D_real, valid) + criterion(D_fake, fake_label)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # -------------------------
            # Train Generator
            # -------------------------
            D_fake = D(fake)
            loss_G = criterion(D_fake, valid)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"[Epoch {epoch}/{epochs}] Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    return model
