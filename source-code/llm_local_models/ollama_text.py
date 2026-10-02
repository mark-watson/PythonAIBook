# ollama_text.py - Basic text generation with a local Ollama model
#
# The simplest example: send a prompt to a local model and print the response.
# No API keys needed — the request stays entirely on your machine.
#
# litellm (https://github.com/BerriAI/litellm) names models as
# "provider/model-name", so this same script talks to a cloud API if MODEL
# becomes "openai/gpt-5.4-nano" or "gemini/gemini-3-flash-preview".
#
# Requirements: uv sync; ollama pull llama3.2:3b
# Run: uv run python ollama_text.py

import litellm

MODEL = "ollama_chat/llama3.2:3b"

response = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": "Briefly explain what a neural network is."}],
)
assert isinstance(response, litellm.ModelResponse), "Expected a non-streaming response"

print(response.choices[0].message.content)
