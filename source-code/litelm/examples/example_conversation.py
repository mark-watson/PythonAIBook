"""Multi-turn conversation with in-process history.

The same pattern as ``llm_local_models/ollama_memory.py`` and
``llm_public_apis/gemini_conversation.py``, but provider independent: the
message list is plain dicts in the order the model sees them, and litelm hands
them to whichever provider the model string names.

Run::

    uv run example_conversation.py
    LITELM_MODEL=gemini/gemini-3-flash-preview uv run example_conversation.py
"""

import os

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/llama3.2:3b")
SYSTEM = (
    "You are a concise technical writing assistant. Keep answers under 3 sentences."
)

QUESTIONS = [
    "What is the capital of France?",
    "What is its population?",  # "its" resolves against the previous answer
    "Name the top 3 tourist attractions there.",
]


class Assistant:
    """A model plus the conversation so far."""

    def __init__(self, model: str, system: str) -> None:
        self.model = model
        self.messages: list[litelm.Message] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        response = litelm.completion(self.model, self.messages)
        reply = response.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def message_count(self) -> int:
        return len(self.messages)


def main() -> None:
    assistant = Assistant(MODEL, SYSTEM)
    for question in QUESTIONS:
        print(f"Q: {question}")
        print(f"A: {assistant.chat(question)}")
        print()
    print(f"(conversation has {assistant.message_count()} messages)")


if __name__ == "__main__":
    main()
