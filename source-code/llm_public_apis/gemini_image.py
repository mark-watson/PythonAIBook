# gemini_image.py - Analyzing an image with Gemini
#
# Demonstrates multimodal input: sending both text and an image to the model.
# The model can describe, analyze, or answer questions about the image content.
#
# Adapted from the Solo_Knowledge_Worker_AI photo_understanding.py example.
#
# Requirements: uv pip install google-genai Pillow
# Environment: export GOOGLE_API_KEY="your-api-key"

import io
import os
from typing import Any
from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Load an image from disk (replace with your own image path)
image = Image.open("photo.jpg")
buf = io.BytesIO()
image.save(buf, format="JPEG")
image_bytes = buf.getvalue()

prompt = "Describe what you see in this image. Be specific about people, objects, and setting."

contents: list[Any] = [
    types.Part(text=prompt),
    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
]

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=contents,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0  # no thinking needed for simple description
        )
    ),
)

print(response.text)
