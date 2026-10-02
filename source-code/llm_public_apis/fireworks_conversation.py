# fireworks_conversation.py - Multi-turn conversation with Fireworks
#
# Demonstrates maintaining conversation history across multiple exchanges.
# Each call sends the full message history so the model can resolve
# references like "its" and "there" that depend on prior context.
#
# Requirements: uv sync
# Environment: export FIREWORKS_API_KEY="your-api-key"
# Run: uv run python fireworks_conversation.py

import litelm

MODEL = "fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash"

messages: list[litelm.Message] = []


def chat(user_message: str) -> str:
    """Send a message and get a response, maintaining conversation history."""
    messages.append({"role": "user", "content": user_message})
    response = litelm.completion(MODEL, messages)
    reply = response.content
    if reply is None:
        raise RuntimeError("Empty response from model")
    messages.append({"role": "assistant", "content": reply})
    return reply


# A multi-turn conversation — note how later questions reference earlier answers
print(chat("What is the capital of France?"))
print(chat("What is its population?"))  # "its" refers to Paris from context
print(chat("What are the top 3 tourist attractions there?"))
