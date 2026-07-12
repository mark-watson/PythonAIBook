# fireworks_text.py - Basic text generation with Fireworks.ai
#
# Demonstrates the simplest possible use of the Fireworks API:
# send a text prompt, receive a generated response.
# Uses the OpenAI-compatible chat completions endpoint.
#
# Requirements: uv pip install openai
# Environment: export FIREWORKS_API_KEY="your-api-key"

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[
        {
            "role": "user",
            "content": "Briefly explain what a transformer model is in AI.",
        }
    ],
)

print(response.choices[0].message.content)
