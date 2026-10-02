# gemini_text.py - Basic text generation with Google Gemini
#
# Demonstrates the simplest possible use of the Gemini API:
# send a text prompt, receive a generated response.
#
# The model string carries the provider prefix, so litellm routes the request to
# Gemini's OpenAI-compatible endpoint. Only that prefix changes to call a
# different provider (see https://github.com/BerriAI/litellm).
#
# Requirements: uv sync
# Environment: export GOOGLE_API_KEY="your-api-key"
# Run: uv run python gemini_text.py

import litellm

MODEL = "gemini/gemini-3-flash-preview"

response = litellm.completion(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Briefly explain what a transformer model is in AI.",
        }
    ],
)

assert isinstance(response, litellm.ModelResponse)
print(response.choices[0].message.content)
