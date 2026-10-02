# ollama_memory.py - Conversation with persistent memory
#
# Demonstrates maintaining conversation history across multiple exchanges.
# The LocalAssistant class keeps a message list that grows over time, giving
# the model context about prior exchanges.
#
# Inspired by the memory examples in "Ollama in Action" but uses a simpler
# in-process approach focused on illustrating the conversation pattern,
# without external vector stores or persistence libraries.
#
# Requirements: uv sync; ollama pull llama3.2:3b
# Run: uv run python ollama_memory.py

from typing import Any

import litellm


class LocalAssistant:
    """A simple conversational assistant that maintains message history."""

    def __init__(
        self, model: str = "ollama_chat/llama3.2:3b", system_prompt: str = ""
    ) -> None:
        self.model = model
        self.messages: list[dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining conversation history."""
        self.messages.append({"role": "user", "content": user_message})
        response = litellm.completion(model=self.model, messages=self.messages)
        assert isinstance(response, litellm.ModelResponse), (
            "Expected a non-streaming response"
        )
        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def message_count(self) -> int:
        """Return the number of messages in the conversation history."""
        return len(self.messages)


# Create an assistant with a specific personality
assistant = LocalAssistant(
    system_prompt="You are a concise technical writing assistant. "
    "Keep answers under 3 sentences."
)

# Multi-turn conversation — the model remembers prior context
print("Q:", "What is gradient descent?")
print("A:", assistant.chat("What is gradient descent?"))
print()

print("Q:", "How does the learning rate affect it?")
print("A:", assistant.chat("How does the learning rate affect it?"))
print()

print("Q:", "What happens if I set it too high?")
print("A:", assistant.chat("What happens if I set it too high?"))
print()

print(f"(Conversation has {assistant.message_count()} messages)")
