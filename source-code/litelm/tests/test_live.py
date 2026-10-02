"""Live tests: they call real providers and are skipped when none is available.

Run the whole suite with a local Ollama server and the relevant keys exported::

    ollama serve
    ollama pull llama3.2:3b
    uv run pytest -q -m live

Everything here is opt-in via the ``live`` marker, so the default ``pytest``
run stays offline and fast.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest

import litelm

pytestmark = pytest.mark.live

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# A quantised local model on a laptop can take minutes over a multi-turn
# context, so the live tests do not rely on the library's default timeout.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))


def ollama_model_available() -> bool:
    """True when a local Ollama server lists the model we want to use."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as response:
            body = response.read().decode("utf-8", "replace")
    # PEP 758 (Python 3.14): no parentheses needed around the tuple.
    except urllib.error.URLError, OSError:
        return False
    return OLLAMA_MODEL.split(":")[0] in body


HAS_OLLAMA = ollama_model_available()
HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY"))
HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
HAS_FIREWORKS = bool(os.environ.get("FIREWORKS_API_KEY"))
HAS_NVIDIA = bool(os.environ.get("NVIDIA_API_KEY"))

needs_ollama = pytest.mark.skipif(
    not HAS_OLLAMA, reason=f"no Ollama server with {OLLAMA_MODEL} at {OLLAMA_HOST}"
)
needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY is not set")
needs_gemini = pytest.mark.skipif(not HAS_GEMINI, reason="no Gemini/Google API key")
needs_fireworks = pytest.mark.skipif(
    not HAS_FIREWORKS, reason="FIREWORKS_API_KEY is not set"
)
needs_nvidia = pytest.mark.skipif(not HAS_NVIDIA, reason="NVIDIA_API_KEY is not set")


@needs_ollama
def test_ollama_completion_and_usage() -> None:
    response = litelm.completion(
        f"ollama/{OLLAMA_MODEL}", "Reply with exactly: ok", timeout=OLLAMA_TIMEOUT
    )
    assert response.content is not None
    assert response.model
    assert response.finish_reason is not None
    assert response.usage is not None
    assert response.usage.total_tokens is not None


@needs_ollama
def test_ollama_streaming() -> None:
    chunks = list(
        litelm.completion(
            f"ollama/{OLLAMA_MODEL}",
            "Count: 1 2 3",
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
    )
    # A thinking model may stream only reasoning, so accept either: what is
    # being verified is that SSE events were framed, decoded and folded.
    assert chunks
    assert "".join(chunk.text for chunk in chunks) or "".join(
        chunk.reasoning for chunk in chunks
    )


@needs_ollama
def test_ollama_accepts_a_multi_turn_history() -> None:
    messages: list[litelm.Message] = [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hello Ada."},
        {"role": "user", "content": "What is my name?"},
    ]
    response = litelm.completion(
        f"ollama/{OLLAMA_MODEL}", messages, timeout=OLLAMA_TIMEOUT
    )
    # Deliberately no assertion on the wording: small local models mis-answer.
    # This checks that a user/assistant/user history is accepted and parsed.
    assert isinstance(response, litelm.Response)
    assert response.finish_reason is not None


@needs_ollama
def test_ollama_tool_calling_round_trip() -> None:
    def get_weather(args: dict[str, object]) -> str:
        return f"sunny and 22C in {args['location']}"

    tool = litelm.make_tool(
        "get_weather",
        "Get the current weather for a location",
        [("location", "string", "City name, e.g. Paris")],
        get_weather,
    )
    model = f"ollama/{OLLAMA_MODEL}"
    messages: list[litelm.Message] = [
        {"role": "user", "content": "What is the weather in Paris? Call get_weather."}
    ]

    first = litelm.completion(model, messages, tools=[tool], timeout=OLLAMA_TIMEOUT)
    if not first.tool_calls:
        pytest.skip(f"{OLLAMA_MODEL} chose not to call the tool for this prompt")
    (call,) = first.tool_calls
    assert call.name == "get_weather"
    assert call.arguments.get("location")

    messages.append(litelm.assistant_message(first))
    for result in litelm.execute_tool_calls([tool], first.tool_calls):
        messages.append(litelm.tool_message(result))

    # Small local models are inconsistent about phrasing the final answer, so
    # this asserts the protocol round-tripped rather than a particular reply.
    second = litelm.completion(model, messages, timeout=OLLAMA_TIMEOUT)
    assert second.finish_reason is not None
    assert second.content is None or isinstance(second.content, str)


@needs_ollama
def test_ollama_chat_with_tools_terminates() -> None:
    def get_weather(args: dict[str, object]) -> str:
        return f"sunny and 22C in {args['location']}"

    tool = litelm.make_tool(
        "get_weather",
        "Get the current weather for a location",
        [("location", "string", "City name, e.g. Paris")],
        get_weather,
    )
    response = litelm.chat_with_tools(
        f"ollama/{OLLAMA_MODEL}",
        "What is the weather in Paris? Call get_weather.",
        [tool],
        max_iterations=4,
        timeout=OLLAMA_TIMEOUT,
    )
    assert isinstance(response, litelm.Response)


@needs_openai
def test_openai_completion() -> None:
    assert litelm.ask("openai/gpt-5.4-nano", "Reply with exactly: ok")


@needs_gemini
def test_gemini_completion() -> None:
    assert litelm.ask("gemini/gemini-3-flash-preview", "Reply with exactly: ok")


@needs_fireworks
def test_fireworks_completion() -> None:
    model = "fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash"
    assert litelm.ask(model, "Reply with exactly: ok")


@needs_nvidia
def test_nvidia_completion() -> None:
    assert litelm.ask("nvidia/meta/llama-3.1-8b-instruct", "Reply with exactly: ok")
