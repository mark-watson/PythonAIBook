# Deep Learning Image Generation – Source Code

This directory contains example code for the **Overview of Image Generation** chapter.

## Running

```bash
uv run image_generation.py
```

Example generated image:


![Generated image](generated_landscape.png)

The Stable Diffusion model weights (~5 GB) are downloaded automatically on first run. A GPU (CUDA or Apple Silicon MPS) is strongly recommended for reasonable generation speed.

## Files

- **image_generation.py** — Text-to-image generation using Stable Diffusion 2.1 via the Hugging Face diffusers library
- **generated_landscape.png** — Sample output image
