Overview

This repository contains my implementation for Assignment 4: Advanced Image Generation.
The goal of the assignment is to add a Diffusion Model and an Energy-Based Model (EBM) to the API previously built in Module 6, and to deploy the updated API using Docker. The repository includes the implementation of the diffusion model, the sampling code, and the FastAPI endpoint. The project is containerized so that it can be built and executed in a stable and reproducible environment.

Project structure
.
├── helper_lib/
│   ├── __init__.py
│   ├── model.py          # Diffusion model (UNet) and sinusoidal time embedding
│   ├── generator.py      # Reverse diffusion sampling
│   └── trainer.py        # Optional training script (not required for the assignment)
│
├── main.py               # FastAPI application exposing the model endpoint
├── Dockerfile            # Docker setup for the API
├── requirements.txt
└── README.md

Diffusion Model

The diffusion model is implemented in helper_lib/model.py.
It includes:

A sinusoidal time embedding function

A small UNet-like architecture for predicting noise

A get_model() function that returns the diffusion model instance

The implementation follows the simplified structure introduced in Module 8.

Sampling

Reverse diffusion sampling is implemented in helper_lib/generator.py.
The sampling function performs a fixed number of diffusion steps and returns generated images.
This implementation is not intended to produce high-quality images, as the assignment does not require training a full diffusion model.

API Endpoint

The main FastAPI application is defined in main.py.
The following endpoint is provided: