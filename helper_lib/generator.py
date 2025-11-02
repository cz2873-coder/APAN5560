import torch
import numpy as np
from PIL import Image

def generate_raw_image(model, device="cpu"):
    """生成单张 28×28 GAN 图像（PIL灰度图像）"""
    model.generator.eval()
    z = torch.randn(1, model.latent_dim).to(device)

    with torch.no_grad():
        img = model.generator(z).cpu()

    # [-1,1] → [0,255]
    img = (img + 1) / 2
    img = img.squeeze().numpy() * 255
    img = img.astype("uint8")

    return Image.fromarray(img)


def generate_grid_image(model, device="cpu", grid_size=3, upscale=3):
    """
    生成 grid_size × grid_size 拼接大图，
    并把图片放大 upscale 倍（默认放大 3 倍）
    """
    # 生成小图
    imgs = [generate_raw_image(model, device=device)
            for _ in range(grid_size * grid_size)]

    w, h = imgs[0].size  # 原始 28×28

    # 创建大图画布
    grid_img = Image.new('L', (w * grid_size, h * grid_size))

    # 将小图粘贴到大图
    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            grid_img.paste(imgs[idx], (col * w, row * h))
            idx += 1

    # ✅ 放大（高质量）
    new_size = (grid_img.width * upscale, grid_img.height * upscale)
    grid_img = grid_img.resize(new_size, Image.NEAREST)

    return grid_img
