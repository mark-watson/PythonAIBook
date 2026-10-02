"""
Image generation using Google's Imagen 4 model
via the Gemini API.

The call goes through litellm's image_generation() entry point: text-to-image has
no OpenAI-compatible endpoint, so litellm reaches the provider's own API, decodes
the returned images, and hands back ImageObject values — no SDK required and
no local GPU or large model downloads.
See https://github.com/BerriAI/litellm.

Requirements:
  uv sync

Set your API key:
  export GOOGLE_API_KEY="your-key-here"
"""

import base64
import os
import urllib.request
from pathlib import Path

import litellm

MODEL = "gemini/imagen-4.0-fast-generate-001"


def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY environment variable")

    prompt = "a serene mountain landscape at sunset, oil painting style"
    print(f"Generating image for prompt: '{prompt}'")

    response = litellm.image_generation(model=MODEL, prompt=prompt, n=1)

    for generated_image in response.data:
        if generated_image.b64_json:
            image_bytes = base64.b64decode(generated_image.b64_json)
        elif generated_image.url:
            with urllib.request.urlopen(generated_image.url) as http_response:
                image_bytes = http_response.read()
        else:
            raise SystemExit("The model returned no image data")

        output_path = Path("gemini_generated_landscape.png")
        output_path.write_bytes(image_bytes)
        print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
