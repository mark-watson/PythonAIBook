"""Text-to-image generation through the uniform interface.

Text-to-image has no OpenAI-compatible endpoint, so
:func:`~litelm.generate_image` reaches the provider's own API -- here Gemini's
Imagen, the model behind
``deep_learning_image_generation/gemini_image_generation.py``. Each result is a
:class:`~litelm.types.GeneratedImage` holding the decoded bytes.

Run::

    export GOOGLE_API_KEY="..."
    uv run example_image.py
    LITELM_MODEL=gemini/imagen-4.0-fast-generate-001 uv run example_image.py
"""

import os
from pathlib import Path

import litelm

MODEL = os.environ.get("LITELM_MODEL", "gemini/imagen-4.0-fast-generate-001")
PROMPT = "a serene mountain landscape at sunset, oil painting style"


def main() -> None:
    print(f"model: {MODEL}")
    print(f"prompt: {PROMPT}")
    images = litelm.generate_image(MODEL, PROMPT, number_of_images=1)
    for index, image in enumerate(images, 1):
        target = image.save(Path(f"generated-{index}.{image.suffix}"))
        print(f"saved {target} ({len(image)} bytes, {image.mime_type})")


if __name__ == "__main__":
    main()
