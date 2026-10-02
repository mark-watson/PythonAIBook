# gemini_text.py - Basic text generation with Google Gemini
#
# Demonstrates the simplest possible use of the Gemini API:
# send a text prompt, receive a generated response.
#
# The model string carries the provider prefix, so litelm routes the request to
# Gemini's OpenAI-compatible endpoint. Only that prefix changes to call a
# different provider (see ../litelm/README.md).
#
# Requirements: uv sync
# Environment: export GOOGLE_API_KEY="your-api-key"
# Run: uv run python gemini_text.py

import litelm

MODEL = "gemini/gemini-3-flash-preview"

response = litelm.completion(
    MODEL,
    messages="Briefly explain what a transformer model is in AI.",
)

print(response.content)
