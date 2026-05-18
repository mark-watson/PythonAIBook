# Deep Learning Image Generation – Source Code

This directory contains example code for the **Overview of Image Generation** chapter.

## Running the Local Stable Diffusion Example

```bash
uv run image_generation.py
```

Example very low-res generated image (prompt: "a serene mountain landscape at sunset, oil painting style")

![Generated image](generated_landscape.png)

The Stable Diffusion model weights (~1.1 GB) are downloaded automatically on first run. A GPU (CUDA or Apple Silicon MPS) is strongly recommended for reasonable generation speed.

## Running the Gemini Imagen API Example

```bash
export GOOGLE_API_KEY="your-key-here"
uv run gemini_image_generation.py
```

This example uses Google's Imagen 4 model via the Gemini API — no local GPU or large model downloads required. The generated image is saved to `gemini_generated_landscape.png`.

Example Gemini-generated image (same prompt):

![Gemini generated image](gemini_generated_landscape.png)

## Files

- **image_generation.py** — Text-to-image generation using Stable Diffusion via the Hugging Face diffusers library (runs locally)
- **gemini_image_generation.py** — Text-to-image generation using Google's Imagen 4 via the Gemini API (cloud-based)
- **generated_landscape.png** — Sample output image (local model)

## Architecture

![Image generation pipeline architecture](FIG_deep_learning_image_generation.jpg)
