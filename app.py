# app.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io
import torch

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_grid_image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

app = FastAPI()
MODEL = None  # 全局 GAN 模型缓存


# ---------------------------
# 加载 MNIST（训练使用）
# ---------------------------
def load_mnist():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.MNIST("data", train=True, download=True, transform= tfm)
    return DataLoader(ds, batch_size=128, shuffle=True)


# ---------------------------
# API 首页
# ---------------------------
@app.get("/")
def home():
    return {"message": "GAN API is running."}


# ---------------------------
# 训练 GAN
# ---------------------------
@app.post("/gan/train")
def train():
    global MODEL
    MODEL = get_model("gan")
    loader = load_mnist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_gan(MODEL, loader, device=device, epochs=10)

    # 保存权重
    torch.save(MODEL.generator.state_dict(), "gan_generator.pth")
    torch.save(MODEL.discriminator.state_dict(), "gan_discriminator.pth")

    return {"status": "training complete", "device": device}


# ---------------------------
# 生成 2×2 PNG 图片
# ---------------------------
@app.get("/gan/generate")
def generate():
    global MODEL

    # 如果模型还没加载
    if MODEL is None:
        MODEL = get_model("gan")

        # 尝试加载训练过的 generator
        try:
            MODEL.generator.load_state_dict(
                torch.load("gan_generator.pth", map_location="cpu")
            )
        except:
            # 如果还没训练，也能生成（效果不好）
            pass

    # 生成 2x2 拼图
    img = generate_grid_image(MODEL, device="cpu", grid_size=3)

    # 转换为 PNG 并返回
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

