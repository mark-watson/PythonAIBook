# openai_text.py - Basic text generation with OpenAI
#
# Demonstrates using the OpenAI Responses API with GPT-5.4-nano. The response
# arrives as an array of typed output items rather than one message, and
# litellm's ResponsesAPIResponse exposes the concatenated text as output_text.
#
# Requirements: uv sync
# Environment: export OPENAI_API_KEY="your-api-key"
# Run: uv run python openai_text.py

import litellm
from litellm.types.utils import ResponsesAPIResponse

MODEL = "openai/gpt-5.4-nano"

response = litellm.responses(
    model=MODEL,
    input="Briefly explain what a transformer model is in AI.",
)

# output_text is the concatenated text of every message item
assert isinstance(response, ResponsesAPIResponse)
print(response.output_text)
