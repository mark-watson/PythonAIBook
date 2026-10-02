"""Value types returned by the litelm entry points.

These are the Python counterparts of the Common Lisp ``response`` struct and
the Racket ``llm-response`` / ``llm-tool-call`` / ``llm-tool-result`` structs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Response", "StreamChunk", "ToolCall", "ToolResult", "Usage"]


@dataclass(frozen=True)
class Usage:
    """Token accounting, when the provider reports it."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ToolCall:
    """A tool call the model asked for -- returned, never executed by litelm.

    ``arguments`` is the decoded JSON object; ``arguments_raw`` keeps the
    original JSON text when the call arrived over the wire, which matters for
    reporting malformed arguments back to the model.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    arguments_raw: str | None = None

    def __str__(self) -> str:
        return f"{self.name}({dict(self.arguments)!r})"


@dataclass(frozen=True)
class ToolResult:
    """What running one ``ToolCall`` produced.

    ``result`` is always a string: handlers that raise, or that are handed
    arguments the tool cannot use, yield an ``"Error: ..."`` string so the
    model can correct itself instead of the program blowing up.
    """

    call_id: str
    name: str
    result: str


@dataclass(frozen=True)
class Response:
    """The result of a chat completion.

    ``content`` is ``None`` when the model only made tool calls. ``reasoning``
    carries a separate reasoning trace when the provider returns one (Fireworks
    ``thinking`` mode, Ollama ``think``); DeepSeek-R1 served through Ollama
    instead leaves its ``<think>`` block inline in ``content``.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    usage: Usage | None = None
    reasoning: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.content or ""


@dataclass(frozen=True)
class StreamChunk:
    """One server-sent event from ``completion(..., stream=True)``.

    ``text`` is the incremental content delta, so the usual loop is::

        for chunk in litelm.completion("ollama/llama3.2:3b", "Count to five.",
                                       stream=True):
            print(chunk.text, end="", flush=True)

    ``tool_calls`` is a *snapshot*: streaming providers send tool calls in
    fragments, and litelm accumulates them, so the final chunk carries the
    complete list. Tool-call fragments arrive before ``finish_reason`` is set.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None
    reasoning: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text
