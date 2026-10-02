# NVIDIA_client.py - Library for NVIDIA's free inference service
#
# Provides helper functions for calling NVIDIA NIM. The free tier gives access
# to a wide catalogue of open models (Llama, Mistral, Phi, DeepSeek, etc.)
# without standing up local GPU hardware.
#
# The endpoint and model id stay exported because the sibling
# ../NVIDIA_Object_Oriented_Agents example builds its own litellm client from
# them; litelm routes the HTTP calls here.
#
# Requirements: uv sync
# Environment: export NVIDIA_API_KEY="your-api-key"
#   Sign up and obtain a free key at: https://build.nvidia.com

import litelm

PROVIDER = "nvidia"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
_BASE_URL = litelm.find_provider(PROVIDER).base_url


def model_id(model: str) -> str:
    """litelm model string for an NVIDIA NIM model id."""
    return f"{PROVIDER}/{model}"


def complete(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Single-turn prompt → reply."""
    return litelm.ask(model_id(model), prompt)


def chat(messages: list[litelm.Message], model: str = DEFAULT_MODEL) -> str:
    """Multi-turn conversation history → next assistant reply."""
    response = litelm.completion(model_id(model), messages)
    content = response.content
    if content is None:
        raise RuntimeError("Empty response from model")
    return content


if __name__ == "__main__":
    print(complete("Briefly explain what a transformer model is in AI."))

    history: list[litelm.Message] = []
    for turn in [
        "What is the capital of France?",
        "What is its population?",
        "Name the top 3 tourist attractions there.",
    ]:
        history.append({"role": "user", "content": turn})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"Q: {turn}\nA: {reply}\n")
