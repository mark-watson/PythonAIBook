# gemini_text.py - Basic text generation with Google Gemini
#
# Demonstrates the simplest possible use of the Gemini API:
# send a text prompt, receive a generated response.
#
# Requirements: uv pip install google-genai
# Environment: export GOOGLE_API_KEY="your-api-key"

import os
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Briefly explain what a transformer model is in AI.",
)

print(response.text)
