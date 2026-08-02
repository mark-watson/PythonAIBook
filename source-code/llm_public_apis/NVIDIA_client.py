# NVIDIA_client.py - Library for NVIDIA's free inference service
#
# Provides helper functions for calling NVIDIA NIM via the
# OpenAI-compatible chat completions endpoint. The free tier gives access
# to a wide catalogue of open models (Llama, Mistral, Phi, DeepSeek, etc.)
# without standing up local GPU hardware.
#
# Requirements: uv pip install openai
# Environment: export NVIDIA_API_KEY="your-api-key"
#   Sign up and obtain a free key at: https://build.nvidia.com

import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
_BASE_URL = "https://integrate.api.nvidia.com/v1"


def get_client() -> OpenAI:
    return OpenAI(
        base_url=_BASE_URL,
        api_key=os.getenv("NVIDIA_API_KEY"),
    )


def complete(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Single-turn prompt → reply."""
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty response from model")
    return content


def chat(
    messages: list[ChatCompletionMessageParam],
    model: str = DEFAULT_MODEL,
) -> str:
    """Multi-turn conversation history → next assistant reply."""
    response = get_client().chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty response from model")
    return content


if __name__ == "__main__":
    print(complete("Briefly explain what a transformer model is in AI."))

    history: list[ChatCompletionMessageParam] = []
    for turn in [
        "What is the capital of France?",
        "What is its population?",
        "Name the top 3 tourist attractions there.",
    ]:
        history.append({"role": "user", "content": turn})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Q: {turn}\nA: {reply}\n")
