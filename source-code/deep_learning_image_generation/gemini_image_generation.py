"""
Image generation using Google's Imagen 4 model
via the Gemini API.

The call goes through litelm's generate_image() entry point: text-to-image has
no OpenAI-compatible endpoint, so litelm reaches the provider's own API, decodes
the returned images, and hands back GeneratedImage values — no SDK required and
no local GPU or large model downloads.

Requirements:
  uv sync

Set your API key:
  export GOOGLE_API_KEY="your-key-here"
"""

import os
from pathlib import Path

import litelm

MODEL = "gemini/imagen-4.0-fast-generate-001"


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY environment variable")

    prompt = "a serene mountain landscape at sunset, oil painting style"
    print(f"Generating image for prompt: '{prompt}'")

    images = litelm.generate_image(MODEL, prompt, number_of_images=1)

    for generated_image in images:
        output_path = Path(f"gemini_generated_landscape.{generated_image.suffix}")
        generated_image.save(output_path)
        print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
