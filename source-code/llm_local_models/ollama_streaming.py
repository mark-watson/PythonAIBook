# ollama_streaming.py - Streaming responses for real-time output
#
# Streaming lets users see output as it's generated, which improves
# perceived responsiveness. Each chunk contains a small piece of text.
#
# Requirements: uv sync; ollama pull llama3.2:3b
# Run: uv run python ollama_streaming.py

import litelm

MODEL = "ollama/llama3.2:3b"

stream = litelm.completion(
    MODEL,
    messages=[{"role": "user", "content": "Write a short poem about programming."}],
    stream=True,
)

# Print each chunk as it arrives, without newlines between chunks
for chunk in stream:
    print(chunk.text, end="", flush=True)
print()  # final newline
