# openai_text.py - Basic text generation with OpenAI
#
# Demonstrates using the OpenAI Responses API with GPT-5.4-nano. The response
# arrives as an array of typed output items rather than one message, and
# litelm.responses folds that array into the usual Response.
#
# Requirements: uv sync
# Environment: export OPENAI_API_KEY="your-api-key"
# Run: uv run python openai_text.py

import litelm

MODEL = "openai/gpt-5.4-nano"

response = litelm.responses(MODEL, "Briefly explain what a transformer model is in AI.")

# content is the concatenated output_text of every message item
print(response.content)
