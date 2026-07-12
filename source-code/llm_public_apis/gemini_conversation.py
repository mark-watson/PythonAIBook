# gemini_conversation.py - Multi-turn conversation with Gemini
#
# Demonstrates maintaining conversation history across multiple exchanges.
# Each call sends the full history so the model can resolve references
# like "its" and "there" that depend on prior context.
#
# Requirements: uv pip install google-genai
# Environment: export GOOGLE_API_KEY="your-api-key"

import os
from typing import Any
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Build a conversation as a list of content parts
conversation: list[Any] = []


def chat(user_message: str) -> str:
    """Send a message and get a response, maintaining conversation history."""
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=conversation
    )
    text = response.text
    assert text is not None
    conversation.append(types.Content(role="model", parts=[types.Part(text=text)]))
    return text


# A multi-turn conversation — note how later questions reference earlier answers
print(chat("What is the capital of France?"))
print(chat("What is its population?"))  # "its" refers to Paris from context
print(chat("What are the top 3 tourist attractions there?"))
