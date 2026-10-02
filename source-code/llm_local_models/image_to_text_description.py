# image_to_text_description.py - Vision model describing a local image
#
# litellm (https://github.com/BerriAI/litellm) takes vision input as
# OpenAI-style content parts. We read the file, sniff its MIME type,
# base64-encode it, and send an image_url part — so the same call works for
# Ollama and for a cloud vision model.
#
# Requirements: uv sync; ollama pull qwen3.5:0.8b (or another vision model)
# Run: uv run python image_to_text_description.py

import base64
import mimetypes
from pathlib import Path

import litellm

# Specify the path to the image file to be analyzed
image_path = Path("ticket.png")

# Inline the image as a data URL the vision models can decode
mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
image_data_url = (
    f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
)

# Send the image to the vision-capable model for a detailed description
response = litellm.completion(
    model="ollama_chat/qwen3.5:0.8b",  # Ensure you use a vision-capable model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ],
    think=False,  # Suppresses the <think> reasoning block
)
assert isinstance(response, litellm.ModelResponse), "Expected a non-streaming response"

# Print the model's descriptive analysis of the image
print(response.choices[0].message.content)
