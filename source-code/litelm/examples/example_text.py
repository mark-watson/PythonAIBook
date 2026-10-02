"""Basic text generation through the uniform interface.

Compare with ``llm_local_models/ollama_text.py`` and
``llm_public_apis/gemini_text.py``: the SDKs differ, the request does not. With
litelm the *only* thing that changes between a local model and a cloud API is
the model string, because every registered provider speaks the same
OpenAI-compatible chat protocol underneath.

Run::

    uv run example_text.py                              # local Ollama
    LITELM_MODEL=openai/gpt-5.4-nano uv run example_text.py
    LITELM_MODEL=gemini/gemini-3-flash-preview uv run example_text.py
    LITELM_MODEL=nvidia/meta/llama-3.1-8b-instruct uv run example_text.py

Any OpenAI-compatible service can be registered at runtime, so this works too::

    litelm.define_provider("groq", "https://api.groq.com/openai/v1",
                           env_keys=["GROQ_API_KEY"])
    litelm.ask("groq/llama-3.1-8b-instant", "What is 2+2?")
"""

import os

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/llama3.2:3b")
PROMPT = "Briefly explain what a neural network is."


def main() -> None:
    response = litelm.completion(MODEL, PROMPT)

    print(f"model:         {response.model}")
    print(f"finish reason: {response.finish_reason}")
    print()
    print(response.content)

    if response.usage is not None:
        usage = response.usage
        print()
        print(
            f"tokens: {usage.prompt_tokens} prompt + "
            f"{usage.completion_tokens} completion = {usage.total_tokens}"
        )


if __name__ == "__main__":
    main()
