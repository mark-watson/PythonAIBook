# fireworks_conversation.py - Multi-turn conversation with Fireworks
#
# Demonstrates maintaining conversation history across multiple exchanges.
# Each call sends the full message history so the model can resolve
# references like "its" and "there" that depend on prior context.
#
# Requirements: uv sync
# Environment: export FIREWORKS_API_KEY="your-api-key"
# Run: uv run python fireworks_conversation.py

from typing import Any

import litellm

MODEL = "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash"

messages: list[dict[str, Any]] = []


def chat(user_message: str) -> str:
    """Send a message and get a response, maintaining conversation history."""
    messages.append({"role": "user", "content": user_message})
    response = litellm.completion(model=MODEL, messages=messages)
    assert isinstance(response, litellm.ModelResponse)
    reply = response.choices[0].message.content
    if reply is None:
        raise RuntimeError("Empty response from model")
    messages.append({"role": "assistant", "content": reply})
    return reply


# A multi-turn conversation — note how later questions reference earlier answers
print(chat("What is the capital of France?"))
print(chat("What is its population?"))  # "its" refers to Paris from context
print(chat("What are the top 3 tourist attractions there?"))
