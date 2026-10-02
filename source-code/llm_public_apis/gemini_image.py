# gemini_image.py - Analyzing an image with Gemini
#
# Demonstrates multimodal input: sending both text and an image to the model.
# The model can describe, analyze, or answer questions about the image content.
#
# litellm takes images as OpenAI-style content parts: the caller base64-encodes
# the bytes and builds the "data:" URL the wire expects.
#
# Adapted from the Solo_Knowledge_Worker_AI photo_understanding.py example.
#
# Requirements: uv sync
# Environment: export GOOGLE_API_KEY="your-api-key"
# Run: uv run python gemini_image.py

import base64
from pathlib import Path

import litellm

MODEL = "gemini/gemini-3-flash-preview"

# Load an image from disk (replace with your own image path)
image_bytes = Path("photo.jpg").read_bytes()

prompt = "Describe what you see in this image. Be specific about people, objects, and setting."

# Build the data URL litellm sends as an image content part
data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"

response = litellm.completion(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ],
    reasoning_effort="minimal",  # no deep thinking for a description
)

assert isinstance(response, litellm.ModelResponse)
print(response.choices[0].message.content)
