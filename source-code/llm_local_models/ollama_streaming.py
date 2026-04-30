# ollama_streaming.py - Streaming responses for real-time output
#
# Streaming lets users see output as it's generated, which improves
# perceived responsiveness. Each chunk contains a small piece of text.
#
# Requirements: ollama pull llama3.2:3b
# Run: uv run ollama_streaming.py

import ollama

stream = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Print each chunk as it arrives, without newlines between chunks
for chunk in stream:
    print(chunk.message.content, end="", flush=True)
print()  # final newline
