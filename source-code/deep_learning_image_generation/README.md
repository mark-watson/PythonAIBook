# Deep Learning Image Generation – Source Code

This directory contains example code for the **Overview of Image Generation** chapter.

## Running

```bash
uv run image_generation.py
```

Example very low-res generated image (prompt: "a serene mountain landscape at sunset, oil painting style")


![Generated image](generated_landscape.png)

The Stable Diffusion model weights (~5 GB) are downloaded automatically on first run. A GPU (CUDA or Apple Silicon MPS) is strongly recommended for reasonable generation speed.

## Files

- **image_generation.py** — Text-to-image generation using Stable Diffusion 2.1 via the Hugging Face diffusers library
- **generated_landscape.png** — Sample output image

## Architecture

![Stable Diffusion image generation pipeline architecture](FIG_deep_learning_image_generation.jpg)
