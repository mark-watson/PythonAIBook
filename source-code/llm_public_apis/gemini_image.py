# gemini_image.py - Analyzing an image with Gemini
#
# Demonstrates multimodal input: sending both text and an image to the model.
# The model can describe, analyze, or answer questions about the image content.
#
# The "images" key on a user message takes a path, a URL, or raw bytes; litelm
# reads the file, sniffs its MIME type, base64-encodes it, and builds the
# content parts the wire expects.
#
# Adapted from the Solo_Knowledge_Worker_AI photo_understanding.py example.
#
# Requirements: uv sync
# Environment: export GOOGLE_API_KEY="your-api-key"
# Run: uv run python gemini_image.py

from pathlib import Path

import litelm

MODEL = "gemini/gemini-3-flash-preview"

# Load an image from disk (replace with your own image path)
image_bytes = Path("photo.jpg").read_bytes()

prompt = "Describe what you see in this image. Be specific about people, objects, and setting."

response = litelm.completion(
    MODEL,
    messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
    extra={"reasoning_effort": "minimal"},  # no deep thinking for a description
)

print(response.content)
