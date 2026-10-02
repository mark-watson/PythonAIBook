# ollama_openai_compat.py - One OpenAI-compatible client for every provider
#
# Ollama exposes an OpenAI-compatible API, and that is exactly the protocol
# litellm speaks. The local server is therefore just another provider: only the
# prefix of the model string decides where the request goes, so the same call
# reaches a cloud API when MODEL changes.
#
# Requirements: uv sync; ollama pull llama3.2:3b
# Run: uv run python ollama_openai_compat.py
#      LLM_MODEL=fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash \
#          uv run python ollama_openai_compat.py

import os

import litellm

# LLM_MODEL rather than LITELLM_MODEL: litellm already owns the LITELLM_*
# environment namespace for its own settings.
MODEL = os.environ.get("LLM_MODEL", "ollama_chat/llama3.2:3b")

response = litellm.completion(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the difference between a list and a tuple in Python?",
        },
    ],
    temperature=0.7,
)
assert isinstance(response, litellm.ModelResponse), "Expected a non-streaming response"

print(f"model: {MODEL}")
print(response.choices[0].message.content)
