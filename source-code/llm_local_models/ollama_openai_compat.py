# ollama_openai_compat.py - Using local Ollama with the OpenAI SDK
#
# Ollama exposes an OpenAI-compatible API, so you can use the standard
# openai Python library to talk to local models. This is useful for
# writing code that can switch between cloud and local models by
# changing only the base URL.
#
# Requirements: ollama pull llama3.2:3b
# Run: uv run ollama_openai_compat.py

from openai import OpenAI

# Point the OpenAI client at the local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed",  # Ollama doesn't require authentication locally
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the difference between a list and a tuple in Python?",
        },
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
