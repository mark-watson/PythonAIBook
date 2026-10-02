"""Shared fixtures.

Every test runs offline: the network layer is replaced by
:class:`FakeTransport`, which records the requests litelm would have sent and
replays canned responses. Assertions therefore cover the exact JSON bodies and
headers, which is where provider bugs actually live.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import litelm


class FakeTransport:
    """Stand-in for :mod:`litelm.transport` that never touches the network."""

    def __init__(self) -> None:
        self.json_results: list[dict[str, Any]] = []
        self.sse_results: list[list[dict[str, Any]]] = []
        self.errors: list[Exception] = []
        self.requests: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float = litelm.DEFAULT_TIMEOUT,
    ) -> dict[str, Any]:
        self._record(url, headers, payload, timeout)
        if self.errors:
            raise self.errors.pop(0)
        if not self.json_results:
            raise AssertionError("FakeTransport ran out of json_results")
        return self.json_results.pop(0)

    def post_sse(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        *,
        timeout: float = litelm.DEFAULT_TIMEOUT,
    ) -> Iterator[dict[str, Any]]:
        self._record(url, headers, payload, timeout)
        if self.errors:
            raise self.errors.pop(0)
        if not self.sse_results:
            raise AssertionError("FakeTransport ran out of sse_results")
        yield from self.sse_results.pop(0)

    @property
    def last_request(self) -> dict[str, Any]:
        assert self.requests, "no request was made"
        return self.requests[-1]

    @property
    def last_payload(self) -> dict[str, Any]:
        payload = self.last_request["payload"]
        assert isinstance(payload, dict)
        return payload

    def _record(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> None:
        self.requests.append(
            {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
        )


@pytest.fixture
def fake_transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    """Replace the HTTP layer for the duration of one test."""
    fake = FakeTransport()
    monkeypatch.setattr(litelm.transport, "post_json", fake.post_json)
    monkeypatch.setattr(litelm.transport, "post_sse", fake.post_sse)
    return fake


@pytest.fixture(autouse=True)
def _clear_api_keys(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep provider key lookups deterministic regardless of the dev machine.

    Tests marked ``live`` are exempt: they exist precisely to use the keys and
    servers the developer has set up.
    """
    if request.node.get_closest_marker("live") is not None:
        return
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "FIREWORKS_API_KEY",
        "NVIDIA_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def chat_body(
    content: str | None = "hello",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "test-model",
    finish_reason: str = "stop",
    usage: dict[str, Any] | None = None,
    reasoning: str | None = None,
) -> dict[str, Any]:
    """Build a minimal OpenAI-compatible chat completion body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage
        if usage is not None
        else {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


def tool_call_body(
    name: str, arguments: str, call_id: str = "call_1"
) -> dict[str, Any]:
    """One wire-format tool call."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def delta(text: str, **extra: Any) -> dict[str, Any]:
    """One streaming event carrying a content delta."""
    choice: dict[str, Any] = {"index": 0, "delta": {"content": text}}
    choice.update(extra)
    return {"model": "test-model", "choices": [choice]}
