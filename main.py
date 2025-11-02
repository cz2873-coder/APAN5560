import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_grid_image


def load_mnist():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=128, shuffle=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    gan = get_model("gan")
    loader = load_mnist()

    # ✅ 训练 GAN
    train_gan(gan, loader, device=device, epochs=10)

    # ✅ 保存模型权重
    torch.save(gan.generator.state_dict(), "gan_generator.pth")
    torch.save(gan.discriminator.state_dict(), "gan_discriminator.pth")
    print("Weights saved.")

    # ✅ 生成 3×3 图片
    grid_img = generate_grid_image(gan, device="cpu", grid_size=3, upscale=3)
    grid_img.save("gan_grid_3x3.png")
    print("Generated 3×3 image saved as gan_grid_3x3.png")


if __name__ == "__main__":
    main()
