# fireworks_text.py - Basic text generation with Fireworks.ai
#
# Demonstrates the simplest possible use of the Fireworks API:
# send a text prompt, receive a generated response.
# Fireworks speaks the OpenAI-compatible chat completions protocol, which is
# the protocol litelm uses for every provider.
#
# Requirements: uv sync
# Environment: export FIREWORKS_API_KEY="your-api-key"
# Run: uv run python fireworks_text.py

import litelm

MODEL = "fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash"

response = litelm.completion(
    MODEL,
    messages=[
        {
            "role": "user",
            "content": "Briefly explain what a transformer model is in AI.",
        }
    ],
)

print(response.content)
