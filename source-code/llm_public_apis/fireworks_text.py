# fireworks_text.py - Basic text generation with Fireworks.ai
#
# Demonstrates the simplest possible use of the Fireworks API:
# send a text prompt, receive a generated response.
# Fireworks speaks the OpenAI-compatible chat completions protocol, which is
# the protocol litellm uses for every provider.
#
# Requirements: uv sync
# Environment: export FIREWORKS_API_KEY="your-api-key"
# Run: uv run python fireworks_text.py

import litellm

MODEL = "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash"

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
