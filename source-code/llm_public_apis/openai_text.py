# openai_text.py - Basic text generation with OpenAI
#
# Demonstrates using the OpenAI Responses API with GPT-5.4-nano.
# The response structure requires iterating through output items
# to extract the assistant's text message.
#
# Requirements: uv pip install openai
# Environment: export OPENAI_API_KEY="your-api-key"

from openai import OpenAI
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

client = OpenAI()  # reads OPENAI_API_KEY from environment

response = client.responses.create(
    model="gpt-5.4-nano", input="Briefly explain what a transformer model is in AI."
)

# Extract the assistant's text from the response output
for item in response.output:
    if isinstance(item, ResponseOutputMessage) and item.role == "assistant":
        for content in item.content:
            if isinstance(content, ResponseOutputText):
                print(content.text)
                break
