"""NVIDIA's free inference service through the uniform interface.

``llm_public_apis/NVIDIA_client.py`` hand-builds an OpenAI client pointed at
``https://integrate.api.nvidia.com/v1`` and wraps it in ``complete`` / ``chat``
helpers. litelm already ships that provider, so those helpers collapse to one
line each -- and they keep working when the model string names a different
provider, which is the whole point of the uniform interface.

Get a free key at https://build.nvidia.com and export it::

    export NVIDIA_API_KEY="nvapi-..."

Run::

    uv run example_nvidia.py
"""

import os

import litelm

MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/meta/llama-3.1-8b-instruct")

TURNS = [
    "What is the capital of France?",
    "What is its population?",
    "Name the top 3 tourist attractions there.",
]


def complete(prompt: str, model: str = MODEL) -> str:
    """Single-turn prompt -> reply."""
    return litelm.ask(model, prompt)


def chat(messages: list[litelm.Message], model: str = MODEL) -> str:
    """Conversation history -> next assistant reply."""
    response = litelm.completion(model, messages)
    if response.content is None:
        raise RuntimeError("Empty response from model")
    return response.content


def main() -> None:
    print(complete("Briefly explain what a transformer model is in AI."))
    print()

    history: list[litelm.Message] = []
    for turn in TURNS:
        history.append({"role": "user", "content": turn})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Q: {turn}\nA: {reply}\n")


if __name__ == "__main__":
    main()
