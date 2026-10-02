# image_to_text_description.py - Vision model describing a local image
#
# litelm reads the file named in the message's "images" key, sniffs its MIME
# type, base64-encodes it, and sends the image_url parts the vision models
# expect — so the same call works for Ollama and for a cloud vision model.
#
# Requirements: uv sync; ollama pull qwen3.5:0.8b (or another vision model)
# Run: uv run python image_to_text_description.py

import litelm

# Specify the path to the image file to be analyzed
image_path = "ticket.png"

# Send the image to the vision-capable model for a detailed description
response = litelm.completion(
    "ollama/qwen3.5:0.8b",  # Ensure you use a vision-capable model
    messages=[
        {
            "role": "user",
            "content": "Describe this image in detail",
            "images": [image_path],
        }
    ],
    extra={"think": False},  # Suppresses the <think> reasoning block
)

# Print the model's descriptive analysis of the image
print(response.content)
