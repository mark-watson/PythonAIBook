# fireworks_conversation.py - Multi-turn conversation with Fireworks
#
# Demonstrates maintaining conversation history across multiple exchanges.
# Each call sends the full message history so the model can resolve
# references like "its" and "there" that depend on prior context.
#
# Requirements: uv pip install openai
# Environment: export FIREWORKS_API_KEY="your-api-key"

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

messages = []

def chat(user_message):
    """Send a message and get a response, maintaining conversation history."""
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="accounts/fireworks/models/deepseek-v4-flash",
        messages=messages,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

# A multi-turn conversation — note how later questions reference earlier answers
print(chat("What is the capital of France?"))
print(chat("What is its population?"))  # "its" refers to Paris from context
print(chat("What are the top 3 tourist attractions there?"))
