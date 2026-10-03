# NVIDIA_client.py - Library for NVIDIA's free inference service
#
# Provides helper functions for calling NVIDIA NIM. The free tier gives access
# to a wide catalogue of open models (Llama, Mistral, Phi, DeepSeek, etc.)
# without standing up local GPU hardware.
#
# The endpoint and model id stay exported because the sibling
# ../NVIDIA_Object_Oriented_Agents example builds its own litellm client from
# them; litellm routes the HTTP calls here.
#
# Requirements: uv sync
# Environment: export NVIDIA_API_KEY="your-api-key"
#   Sign up and obtain a free key at: https://build.nvidia.com

import os
from typing import Any

import litellm

PROVIDER = "nvidia_nim"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
_BASE_URL = "https://integrate.api.nvidia.com/v1"


def model_id(model: str) -> str:
    """litellm model string for an NVIDIA NIM model id."""
    return f"{PROVIDER}/{model}"


def complete(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Single-turn prompt → reply."""
    response = litellm.completion(
        model=model_id(model),
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    assert isinstance(response, litellm.ModelResponse)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty response from model")
    return content


def chat(messages: list[dict[str, Any]], model: str = DEFAULT_MODEL) -> str:
    """Multi-turn conversation history → next assistant reply."""
    response = litellm.completion(
        model=model_id(model),
        messages=messages,
        api_key=os.getenv("NVIDIA_API_KEY"),
    )
    assert isinstance(response, litellm.ModelResponse)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty response from model")
    return content


if __name__ == "__main__":
    print(complete("Briefly explain what a transformer model is in AI."))

    history: list[dict[str, Any]] = []
    for turn in [
        "What is the capital of France?",
        "What is its population?",
        "Name the top 3 tourist attractions there.",
    ]:
        history.append({"role": "user", "content": turn})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Q: {turn}\nA: {reply}\n")
