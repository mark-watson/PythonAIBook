"""
Image generation using Google's Imagen 4 model
via the Gemini API.

Uses the google-genai SDK for a simple, low-code
approach to text-to-image generation — no local GPU
or large model downloads required.

Requirements:
  uv add google-genai Pillow

Set your API key:
  export GOOGLE_API_KEY="your-key-here"
"""

import io
import os

from google import genai
from google.genai import types
from PIL import Image


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY environment variable")

    client = genai.Client(api_key=api_key)

    prompt = "a serene mountain landscape at sunset, oil painting style"
    print(f"Generating image for prompt: '{prompt}'")

    response = client.models.generate_images(
        model="imagen-4.0-fast-generate-001",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
        ),
    )

    for generated_image in response.generated_images:
        image = Image.open(io.BytesIO(generated_image.image.image_bytes))
        output_path = "gemini_generated_landscape.png"
        image.save(output_path)
        print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
