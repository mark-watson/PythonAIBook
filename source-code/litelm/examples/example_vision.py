"""Image understanding through the uniform interface.

Attach local files (or URLs, or raw bytes) with the ``images`` key on a user
message and litelm turns them into the ``image_url`` content parts an
OpenAI-compatible API expects, base64-encoding the file and sniffing its MIME
type. The same call therefore works for the local Ollama vision model in
``llm_local_models/image_to_text_description.py`` and for
``llm_public_apis/gemini_image.py``.

Run::

    ollama pull qwen3.5:0.8b
    uv run example_vision.py ticket.png
    LITELM_MODEL=gemini/gemini-3-flash-preview uv run example_vision.py photo.jpg
"""

import os
import sys
from typing import Any

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/qwen3.5:0.8b")
PROMPT = "Describe this image in detail."


def provider_extras(model: str) -> dict[str, Any] | None:
    """Provider-specific request fields go through ``extra``.

    Ollama's vision models emit a ``<think>`` reasoning block unless asked not
    to; other providers reject the unknown field, so it is sent only locally.
    """
    if model.startswith("ollama/"):
        return {"think": False}
    return None


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else "ticket.png"
    messages = [
        {"role": "user", "content": PROMPT, "images": [image_path]},
    ]

    response = litelm.completion(MODEL, messages, extra=provider_extras(MODEL))
    print(f"model: {MODEL}")
    print(f"image: {image_path}\n")
    print(response.content)


if __name__ == "__main__":
    main()
