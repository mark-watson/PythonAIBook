# gemini_conversation.py - Multi-turn conversation with Gemini
#
# Demonstrates maintaining conversation history across multiple exchanges.
# Each call sends the full history so the model can resolve references
# like "its" and "there" that depend on prior context.
#
# Requirements: uv sync
# Environment: export GOOGLE_API_KEY="your-api-key"
# Run: uv run python gemini_conversation.py

import litelm

MODEL = "gemini/gemini-3-flash-preview"

# Build a conversation as a list of messages
conversation: list[litelm.Message] = []


def chat(user_message: str) -> str:
    """Send a message and get a response, maintaining conversation history."""
    conversation.append({"role": "user", "content": user_message})
    response = litelm.completion(MODEL, conversation)
    text = response.content
    if text is None:
        raise RuntimeError("Empty response from model")
    conversation.append({"role": "assistant", "content": text})
    return text


# A multi-turn conversation — note how later questions reference earlier answers
print(chat("What is the capital of France?"))
print(chat("What is its population?"))  # "its" refers to Paris from context
print(chat("What are the top 3 tourist attractions there?"))
