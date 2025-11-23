from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch

from helper_lib.model import get_model
from helper_lib.generator import generate_samples


app = FastAPI(title="Advanced Image Generation API")

# Optional CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiffusionGenerateRequest(BaseModel):
    num_samples: int = 4
    diffusion_steps: int = 1000
    image_size: int = 28
    seed: Optional[int] = None


class DiffusionGenerateResponse(BaseModel):
    # images[num_samples][H][W]
    images: list[list[list[float]]]


@app.get("/health")
def health_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "device": device}


@app.post("/generate/diffusion", response_model=DiffusionGenerateResponse)
def generate_diffusion_endpoint(request: DiffusionGenerateRequest):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate diffusion model
    model = get_model("Diffusion")

    # NOTE: In a full solution you would typically load trained weights:
    # model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))

    samples = generate_samples(
        model=model,
        device=device,
        num_samples=request.num_samples,
        diffusion_steps=request.diffusion_steps,
        image_size=request.image_size,
        plot=False,
        seed=request.seed,
    )

    imgs = samples.squeeze(1).cpu().tolist()
    return DiffusionGenerateResponse(images=imgs)
