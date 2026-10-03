# image_to_text_description.py - Vision model describing a local image
#
# This demo uses Ollama's own Python SDK rather than litellm. Ollama takes an
# image as the "images" field on a message — a path or raw bytes — and decodes
# it for you, which is simpler than assembling OpenAI-style image_url content
# parts by hand. The rest of the chapter still goes through litellm.
#
# Requirements: uv sync; ollama pull qwen3.5:0.8b (or another vision model)
# Run: uv run python image_to_text_description.py

import ollama

# Specify the path to the image file to be analyzed
image_path = "ticket.png"

# Send the image to the vision-capable model for a detailed description
response = ollama.chat(
    model="qwen3.5:0.8b",  # Ensure you use a vision-capable model
    messages=[
        {
            "role": "user",
            "content": "Describe this image in detail",
            "images": [image_path],
        }
    ],
    think=False,  # Suppresses the <think> reasoning block
)

# Print the model's descriptive analysis of the image
print(response.message.content)
