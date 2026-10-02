# ollama_openai_compat.py - One OpenAI-compatible client for every provider
#
# Ollama exposes an OpenAI-compatible API, and that is exactly the protocol
# litelm speaks. The local server is therefore just another provider: only the
# prefix of the model string decides where the request goes, so the same call
# reaches a cloud API when MODEL changes.
#
# Requirements: uv sync; ollama pull llama3.2:3b
# Run: uv run python ollama_openai_compat.py
#      LITELM_MODEL=fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash \
#          uv run python ollama_openai_compat.py

import os

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/llama3.2:3b")

response = litelm.completion(
    MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the difference between a list and a tuple in Python?",
        },
    ],
    temperature=0.7,
)

print(f"model: {MODEL}")
print(response.content)
