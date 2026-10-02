"""Tools: plain Python functions the model is allowed to call.

The Racket port of litelm lets tools be Racket procedures; the Python version
lets them be ordinary Python callables taking one ``dict`` of arguments. Wrap
one and hand it to :func:`litelm.completion`::

    def get_weather(args: dict[str, Any]) -> str:
        return f"sunny and 22C in {args['location']}"

    weather = litelm.make_tool(
        "get_weather",
        "Get the current weather for a location",
        [("location", "string", "City name, e.g. Paris")],
        get_weather,
    )

    litelm.ask("ollama/llama3.2:3b", "Weather in Paris?", tools=[weather])

Tool calls are never executed by ``completion``: it *returns* them. Call
:func:`execute_tool_calls` yourself, or let :func:`litelm.chat_with_tools` run
the whole request / execute / reply loop.

Failures -- unknown tool, malformed JSON arguments, a missing required
argument, an exception inside the handler -- become ``"Error: ..."`` result
strings for the model to read, never uncaught exceptions in your program.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import LitelmError
from .types import ToolCall, ToolResult

__all__ = [
    "Handler",
    "Tool",
    "ToolParam",
    "execute_tool_calls",
    "make_tool",
    "normalize_tools",
    "param",
    "tool_schemas",
]

#: A tool handler takes the decoded JSON arguments and returns anything
#: printable; ``None`` is reported to the model as an empty result.
Handler = Callable[[dict[str, Any]], Any]

#: Shorthand accepted by :func:`make_tool`: ``(name, type, description)``.
ParamTuple = tuple[str, str, str]


@dataclass(frozen=True)
class ToolParam:
    """One entry of a tool's JSON-schema ``properties`` object."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    enum: tuple[str, ...] | None = None


def param(
    name: str,
    type: str,
    description: str,
    *,
    required: bool = True,
    enum: Sequence[str] | None = None,
) -> ToolParam:
    """Build a :class:`ToolParam`; the terse form is a 3-tuple in ``make_tool``."""
    if not isinstance(name, str) or not name:
        raise LitelmError(f"tool parameter name must be a non-empty string: {name!r}")
    if not isinstance(description, str):
        raise LitelmError(
            f"tool parameter description must be a string: {description!r}"
        )
    return ToolParam(
        name=name,
        type=type,
        description=description,
        required=required,
        enum=tuple(enum) if enum is not None else None,
    )


@dataclass
class Tool:
    """A function the model may ask litelm to run.

    ``parameters`` holds :func:`param` values. Use :func:`make_tool` if you
    would rather write parameters as ``(name, type, description)`` tuples.
    """

    name: str
    description: str
    parameters: Sequence[ToolParam]
    func: Handler

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise LitelmError(f"tool name must be a non-empty string: {self.name!r}")
        if not isinstance(self.description, str):
            raise LitelmError(
                f"tool description must be a string: {self.description!r}"
            )
        if not callable(self.func):
            raise LitelmError(f"tool handler must be callable: {self.func!r}")
        for spec in self.parameters:
            if not isinstance(spec, ToolParam):
                raise LitelmError(
                    "tool parameters must be litelm.param(...) values, or "
                    f"(name, type, description) tuples passed to make_tool: {spec!r}"
                )


def make_tool(
    name: str,
    description: str,
    parameters: Sequence[ToolParam | ParamTuple] = (),
    func: Handler | None = None,
) -> Tool:
    """Build a :class:`Tool`, accepting 3-tuples for simple parameters.

    ``parameters`` entries are either ``(name, type, description)`` -- required
    by default -- or :func:`param` values when a parameter is optional or has
    an ``enum``.
    """
    if func is None:
        raise LitelmError(f"tool {name!r} needs a handler function")
    return Tool(
        name=name,
        description=description,
        parameters=[_coerce_param(spec) for spec in parameters],
        func=func,
    )


def normalize_tools(tools: Sequence[Tool] | Mapping[str, Tool] | None) -> list[Tool]:
    """Accept a list of tools or a ``name -> Tool`` mapping; return a list."""
    if tools is None:
        return []
    if isinstance(tools, Mapping):
        return list(tools.values())
    if isinstance(tools, Sequence):
        out: list[Tool] = []
        for tool in tools:
            if not isinstance(tool, Tool):
                raise LitelmError(f"tools must contain Tool values: {tool!r}")
            out.append(tool)
        return out
    raise LitelmError(f"tools must be a list or name -> Tool mapping: {tools!r}")


def tool_schemas(
    tools: Sequence[Tool] | Mapping[str, Tool] | None,
) -> list[dict[str, Any]]:
    """Render tools as the ``tools`` array of an OpenAI-compatible request."""
    schemas: list[dict[str, Any]] = []
    for tool in normalize_tools(tools):
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p in tool.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = list(p.enum)
            properties[p.name] = prop
            if p.required:
                required.append(p.name)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return schemas


def execute_tool_calls(
    tools: Sequence[Tool] | Mapping[str, Tool],
    tool_calls: Sequence[ToolCall],
) -> list[ToolResult]:
    """Run ``tool_calls`` against ``tools`` and return one result per call."""
    registry: dict[str, Tool] = (
        dict(tools)
        if isinstance(tools, Mapping)
        else {t.name: t for t in normalize_tools(tools)}
    )
    return [_run_one(registry, call) for call in tool_calls]


def _run_one(registry: Mapping[str, Tool], call: ToolCall) -> ToolResult:
    if not isinstance(call, ToolCall):
        raise LitelmError(f"want a ToolCall: {call!r}")
    return ToolResult(
        call_id=call.id,
        name=call.name,
        result=_invoke(registry.get(call.name), call),
    )


def _invoke(tool: Tool | None, call: ToolCall) -> str:
    if tool is None:
        return f"Error: unknown tool: {call.name}"

    if call.arguments_raw is not None:
        try:
            decoded = json.loads(call.arguments_raw)
        except json.JSONDecodeError:
            decoded = None
        if not isinstance(decoded, dict):
            return (
                f"Error: invalid JSON arguments for tool {call.name!r}. "
                f"Received: {call.arguments_raw}"
            )

    missing = [
        p.name for p in tool.parameters if p.required and p.name not in call.arguments
    ]
    if missing:
        return (
            f"Error: tool {call.name!r} missing required argument(s): "
            f"{', '.join(missing)}"
        )

    try:
        value = tool.func(dict(call.arguments))
    except Exception as exc:  # noqa: BLE001 -- reported to the model, not raised
        return f"Error: tool {call.name!r} raised: {exc}"
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _coerce_param(spec: ToolParam | ParamTuple) -> ToolParam:
    if isinstance(spec, ToolParam):
        return spec
    if isinstance(spec, tuple) and len(spec) == 3:
        name, type_, description = spec
        return param(name, type_, description)
    raise LitelmError(
        "bad tool parameter spec (want (name, type, description) or "
        f"litelm.param(...)): {spec!r}"
    )
