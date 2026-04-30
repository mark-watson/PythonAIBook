# ollama_text.py - Basic text generation with a local Ollama model
#
# The simplest example: send a prompt to a local model and print the response.
# No API keys needed — the request stays entirely on your machine.
#
# Requirements: ollama pull llama3.2:3b
# Run: uv run ollama_text.py

import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "Briefly explain what a neural network is."}
    ]
)

print(response.message.content)
