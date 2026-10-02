"""Message and image handling for the uniform chat interface.

Python already writes chat messages in the shape the wire wants, so unlike the
Common Lisp and Racket versions of litelm there is no separate user-level
syntax to translate. What this module does provide is:

* ``normalize_messages`` -- accept a bare string, a single dict, or a list of
  either, and validate the result, so a bad role or a tool message without its
  ``tool_call_id`` fails with a clear ``LitelmError`` instead of a provider 400.
* ``assistant_message`` / ``tool_message`` -- build the two continuation
  messages a tool loop needs.
* image attachment -- ``{"role": "user", "content": ..., "images": [...]}`` is
  turned into OpenAI-style ``image_url`` content parts, with local files read
  and base64-encoded, so the same call works for Ollama, Gemini, OpenAI,
  Fireworks and NVIDIA.

::

    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is in this picture?",
         "images": ["ticket.png"]},
    ]
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import LitelmError
from .types import Response, ToolCall, ToolResult

__all__ = [
    "Message",
    "assistant_message",
    "normalize_messages",
    "tool_message",
]

#: Anything litelm accepts where messages are wanted.
Message = str | Mapping[str, Any]
Messages = str | Mapping[str, Any] | Sequence[Message]

VALID_ROLES = ("system", "user", "assistant", "tool")

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def normalize_messages(messages: Messages) -> list[dict[str, Any]]:
    """Return wire-ready message dicts.

    Accepts a string (shorthand for one user message), a single message dict,
    or a sequence of either.
    """
    if isinstance(messages, str):
        return [_normalize_one({"role": "user", "content": messages})]
    if isinstance(messages, Mapping):
        return [_normalize_one(messages)]
    if isinstance(messages, Sequence):
        return [_normalize_one(item) for item in messages]
    raise LitelmError(
        f"messages must be a string, a dict, or a list of them: {messages!r}"
    )


def assistant_message(response: Response) -> dict[str, Any]:
    """Build the assistant continuation message for ``response``.

    Pair it with :func:`tool_message` to drive the request/execute/reply loop by
    hand instead of calling :func:`litelm.chat_with_tools`.

    A turn that produced tool calls but no text omits ``content`` entirely
    rather than sending an empty string: several servers (and the models behind
    them) treat ``"content": ""`` next to ``tool_calls`` as a finished, empty
    answer and stop.
    """
    if not isinstance(response, Response):
        raise LitelmError(f"want a Response: {response!r}")
    message: dict[str, Any] = {"role": "assistant"}
    if response.tool_calls:
        message["tool_calls"] = [
            tool_call_to_wire(call) for call in response.tool_calls
        ]
        if response.content:
            message["content"] = response.content
    elif response.content is not None:
        message["content"] = response.content
    return message


def tool_message(result: ToolResult) -> dict[str, Any]:
    """Build the ``role: "tool"`` message carrying one tool result."""
    if not isinstance(result, ToolResult):
        raise LitelmError(f"want a ToolResult: {result!r}")
    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": result.result,
    }


def tool_call_to_wire(call: ToolCall) -> dict[str, Any]:
    """Render a :class:`~litelm.types.ToolCall` the way the wire wants it."""
    arguments = call.arguments_raw
    if arguments is None:
        arguments = json.dumps(dict(call.arguments))
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": arguments},
    }


def _normalize_one(message: Message) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise LitelmError(
            "each message must be a dict with a 'role' key "
            f"(or a plain string): {message!r}"
        )

    role = _role_string(message.get("role"))
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise LitelmError(f"message content must be a string or None: {message!r}")

    tool_calls = _tool_calls(message.get("tool_calls"))
    tool_call_id = message.get("tool_call_id")
    images = list(_image_sources(message.get("images")))
    name = message.get("name")

    if role == "tool" and not tool_call_id:
        raise LitelmError(f"role 'tool' messages need a tool_call_id: {message!r}")
    if role == "assistant" and content is None and not tool_calls:
        raise LitelmError(f"assistant messages need content or tool_calls: {message!r}")
    if images and role != "user":
        raise LitelmError(f"only user messages can carry images: {message!r}")

    wire: dict[str, Any] = {"role": role}
    if images:
        wire["content"] = _content_parts(content, images)
    elif content is not None:
        wire["content"] = content
    if tool_calls:
        wire["tool_calls"] = tool_calls
    if tool_call_id:
        wire["tool_call_id"] = str(tool_call_id)
    if name:
        wire["name"] = str(name)
    return wire


def _role_string(role: Any) -> str:
    if not isinstance(role, str):
        raise LitelmError(f"message role must be a string: {role!r}")
    lowered = role.strip().lower()
    if lowered not in VALID_ROLES:
        raise LitelmError(
            f"unknown message role {role!r} (expected one of {', '.join(VALID_ROLES)})"
        )
    return lowered


def _tool_calls(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (ToolCall, Mapping)):
        items: Iterable[Any] = [value]
    elif isinstance(value, Sequence):
        items = value
    else:
        raise LitelmError(f"tool_calls must be a list: {value!r}")
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, ToolCall):
            out.append(tool_call_to_wire(item))
        elif isinstance(item, Mapping):
            out.append(dict(item))
        else:
            raise LitelmError(f"tool_calls entries must be ToolCall or dict: {item!r}")
    return out


def _image_sources(value: Any) -> Iterable[str]:
    """Yield data URLs / http URLs for each entry in a message's ``images``."""
    if value is None:
        return []
    if isinstance(value, (str, bytes, os.PathLike)):
        entries: Sequence[Any] = [value]
    elif isinstance(value, Sequence):
        entries = value
    else:
        raise LitelmError(f"images must be a list of paths, bytes, or URLs: {value!r}")
    return [_image_url(entry) for entry in entries]


def _image_url(source: Any) -> str:
    """Turn a path, ``bytes``, or URL into something an OpenAI-compatible API eats."""
    if isinstance(source, bytes):
        return _data_url(source)
    if isinstance(source, os.PathLike):
        return _data_url(Path(source).read_bytes())
    if isinstance(source, str):
        if source.startswith(("http://", "https://", "data:")):
            return source
        return _data_url(Path(source).read_bytes())
    raise LitelmError(f"image must be a file path, bytes, or URL: {source!r}")


def _data_url(data: bytes) -> str:
    return f"data:{_mime_type(data)};base64,{base64.b64encode(data).decode('ascii')}"


def _mime_type(data: bytes) -> str:
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _content_parts(content: str | None, images: Sequence[str]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if content:
        parts.append({"type": "text", "text": content})
    for url in images:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts
