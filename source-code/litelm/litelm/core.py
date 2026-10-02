"""The uniform entry points: ``completion``, ``ask``, ``embedding``,
``chat_with_tools``.

Models are addressed as ``"provider/model-name"`` strings, exactly as in the
Common Lisp ``litelm`` and the Racket ``llmapis`` port. Every registered
provider speaks the OpenAI-compatible chat protocol, so a single code path
handles them all::

    import litelm

    reply = litelm.ask("ollama/llama3.2:3b", "What is 2+2?")
    reply = litelm.ask("openai/gpt-5.4-nano", "What is 2+2?")
    reply = litelm.ask("nvidia/meta/llama-3.1-8b-instruct", "What is 2+2?")

Provider-specific knobs travel in ``extra`` and are merged into the request
body last -- Fireworks/DeepSeek thinking mode, for instance::

    litelm.completion("fireworks-ai/accounts/fireworks/models/deepseek-v4-flash",
                      "Solve the fox, chicken and grain puzzle.",
                      extra={"thinking": {"type": "enabled"}})
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal, overload

from . import transport
from .errors import ApiError, LitelmError
from .messages import Messages, assistant_message, normalize_messages, tool_message
from .providers import (
    bearer_headers,
    parse_model,
    provider_api_key,
    provider_url,
)
from .tools import (
    Tool,
    execute_tool_calls,
    tool_schemas,
)
from .types import Response, StreamChunk, ToolCall, Usage

__all__ = [
    "ask",
    "build_chat_payload",
    "chat_with_tools",
    "completion",
    "embedding",
    "parse_chat_response",
]

#: Tools may be a list or a ``name -> Tool`` mapping.
Tools = Sequence[Tool] | Mapping[str, Tool]
#: ``tool_choice`` values every OpenAI-compatible provider understands.
ToolChoice = str | dict[str, Any] | None


# --------------------------------------------------------------------------
# Request building and response parsing (pure functions, exported for tests)
# --------------------------------------------------------------------------


def build_chat_payload(
    model_name: str,
    messages: Sequence[dict[str, Any]],
    *,
    tools: Tools | None = None,
    tool_choice: ToolChoice = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stream: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON body for an OpenAI-compatible chat request."""
    payload: dict[str, Any] = {"model": model_name, "messages": list(messages)}
    schemas = tool_schemas(tools)
    if schemas:
        payload["tools"] = schemas
        if tool_choice is not None:
            payload["tool_choice"] = _tool_choice(tool_choice)
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    if stream:
        payload["stream"] = True
    if extra:
        payload.update(extra)
    return payload


def parse_chat_response(data: Mapping[str, Any], model_name: str) -> Response:
    """Turn a ``/chat/completions`` body into a :class:`~litelm.types.Response`."""
    if "error" in data:
        raise LitelmError(f"LLM API error: {data['error']}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LitelmError(f"LLM API response has no choices: {_short(data)}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LitelmError(f"LLM API response has a malformed choice: {_short(choice)}")
    raw_message = choice.get("message")
    message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
    return Response(
        content=_text_of(message.get("content")),
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
        finish_reason=_opt_str(choice.get("finish_reason")),
        model=_opt_str(data.get("model")) or model_name,
        usage=_parse_usage(data.get("usage")),
        reasoning=_reasoning_of(message),
        raw=data,
    )


# --------------------------------------------------------------------------
# completion
# --------------------------------------------------------------------------


@overload
def completion(
    model: str,
    messages: Messages,
    *,
    tools: Tools | None = None,
    tool_choice: ToolChoice = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    system: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
    stream: Literal[False] = False,
) -> Response: ...


@overload
def completion(
    model: str,
    messages: Messages,
    *,
    tools: Tools | None = None,
    tool_choice: ToolChoice = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    system: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
    stream: Literal[True],
) -> Iterator[StreamChunk]: ...


def completion(
    model: str,
    messages: Messages,
    *,
    tools: Tools | None = None,
    tool_choice: ToolChoice = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    system: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
    stream: bool = False,
) -> Response | Iterator[StreamChunk]:
    """Send a chat completion to ``model``, a ``"provider/model-name"`` string.

    ``messages`` is a string, a message dict, or a list of either. ``tools`` is
    a list (or ``name -> Tool`` mapping) of tools the model may *ask* to run --
    the calls come back in ``response.tool_calls``, they are never executed
    here. Pass ``stream=True`` to get an iterator of
    :class:`~litelm.types.StreamChunk` instead of a
    :class:`~litelm.types.Response`.

    ``system`` prepends a system message; ``api_key`` / ``api_base`` override
    the provider's defaults for this call; ``extra`` is merged into the request
    body for provider-specific options.
    """
    prov, model_name = parse_model(model, provider)
    key = provider_api_key(prov, api_key)
    headers = bearer_headers(key)
    if extra_headers:
        headers.update(extra_headers)
    url = provider_url(prov, "/chat/completions", api_base)

    wire = normalize_messages(messages)
    if system is not None:
        wire.insert(0, {"role": "system", "content": system})

    payload = build_chat_payload(
        model_name,
        wire,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
        extra=extra,
    )
    if stream:
        events = _stream_events(url, headers, payload, timeout)
        return _stream_chunks(events, model_name)
    return _post_chat(url, headers, payload, model_name, timeout)


def ask(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    tools: Tools | None = None,
    tool_choice: ToolChoice = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
) -> str:
    """One-shot question; returns the reply text (``""`` if the model only
    answered with tool calls)."""
    response = completion(
        model,
        prompt,
        system=system,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        provider=provider,
        api_key=api_key,
        api_base=api_base,
        extra_headers=extra_headers,
        extra=extra,
        timeout=timeout,
        stream=False,
    )
    return response.content or ""


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------


def embedding(
    model: str,
    input: str | Sequence[str],
    *,
    dimensions: int | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
) -> list[list[float]]:
    """Embed ``input`` (a string or list of strings); returns one vector each.

    ``dimensions`` asks providers that support it (OpenAI's
    ``text-embedding-3-*``) for a narrower vector.
    """
    prov, model_name = parse_model(model, provider)
    inputs = _embedding_inputs(input)
    key = provider_api_key(prov, api_key)
    headers = bearer_headers(key)
    if extra_headers:
        headers.update(extra_headers)

    payload: dict[str, Any] = {"model": model_name, "input": inputs}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    if extra:
        payload.update(extra)

    data = transport.post_json(
        provider_url(prov, "/embeddings", api_base), headers, payload, timeout=timeout
    )
    items = data.get("data")
    if not isinstance(items, list):
        raise LitelmError(f"Embeddings response has no data array: {_short(data)}")
    return [_vector_of(item) for item in items]


# --------------------------------------------------------------------------
# the agentic tool loop
# --------------------------------------------------------------------------


def chat_with_tools(
    model: str,
    messages: Messages,
    tools: Tools,
    *,
    max_iterations: int = 10,
    tool_choice: ToolChoice = "auto",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    system: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
) -> Response:
    """Ask ``model`` with ``tools``, run what it asks for, feed the results
    back, and repeat until it answers without tool calls.

    Returns the final response, or the last one seen if ``max_iterations`` runs
    out. For manual control, loop :func:`completion` with
    :func:`litelm.assistant_message` and :func:`litelm.tool_message`.
    """
    if max_iterations < 1:
        raise LitelmError(f"max_iterations must be at least 1: {max_iterations}")

    conversation = normalize_messages(messages)
    remaining = max_iterations
    while True:
        response = completion(
            model,
            conversation,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system=system,
            provider=provider,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
            extra=extra,
            timeout=timeout,
            stream=False,
        )
        if not response.tool_calls or remaining <= 1:
            return response
        remaining -= 1
        conversation.append(assistant_message(response))
        for result in execute_tool_calls(tools, response.tool_calls):
            conversation.append(tool_message(result))


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


def _post_chat(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    model_name: str,
    timeout: float,
) -> Response:
    """POST a chat request, retrying once when OpenAI demands the newer
    ``max_completion_tokens`` spelling of ``max_tokens``."""
    try:
        data = transport.post_json(url, headers, payload, timeout=timeout)
    except ApiError as exc:
        if not _wants_max_completion_tokens(exc, payload):
            raise
        data = transport.post_json(
            url, headers, _rename_max_tokens(payload), timeout=timeout
        )
    return parse_chat_response(data, model_name)


def _stream_events(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> Iterator[dict[str, Any]]:
    """Open the SSE stream, with the same ``max_completion_tokens`` retry as
    the non-streaming path.

    ``transport.post_sse`` is a generator, so the request is only sent when the
    first event is pulled -- which is why the retry wraps that first pull.
    """
    try:
        events = transport.post_sse(url, headers, payload, timeout=timeout)
        first = next(events, None)
    except ApiError as exc:
        if not _wants_max_completion_tokens(exc, payload):
            raise
        events = transport.post_sse(
            url, headers, _rename_max_tokens(payload), timeout=timeout
        )
        first = next(events, None)
    if first is not None:
        yield first
    yield from events


def _rename_max_tokens(payload: Mapping[str, Any]) -> dict[str, Any]:
    renamed = dict(payload)
    renamed["max_completion_tokens"] = renamed.pop("max_tokens")
    return renamed


def _wants_max_completion_tokens(exc: ApiError, payload: Mapping[str, Any]) -> bool:
    return (
        exc.status == 400
        and "max_tokens" in payload
        and "max_completion_tokens" in exc.body
    )


def _stream_chunks(
    events: Iterable[Mapping[str, Any]], model_name: str
) -> Iterator[StreamChunk]:
    """Fold server-sent events into :class:`~litelm.types.StreamChunk` values.

    Content arrives as deltas; tool calls arrive as fragments keyed by
    ``index``, accumulated here so every chunk carries the complete set so far.
    """
    accumulator: dict[int, dict[str, str]] = {}
    for event in events:
        if "error" in event:
            # Some providers report mid-stream failures as an event instead of
            # an HTTP status; do not let the iterator end silently.
            raise LitelmError(f"LLM API error: {event['error']}")
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        raw_delta = choice.get("delta")
        if not isinstance(raw_delta, dict):
            raw_delta = choice.get("message")
        delta: dict[str, Any] = raw_delta if isinstance(raw_delta, dict) else {}
        _accumulate_tool_calls(accumulator, delta.get("tool_calls"))
        yield StreamChunk(
            text=_text_of(delta.get("content")) or "",
            tool_calls=_snapshot_tool_calls(accumulator),
            finish_reason=_opt_str(choice.get("finish_reason")),
            model=_opt_str(event.get("model")) or model_name,
            reasoning=_reasoning_of(delta) or "",
            raw=event,
        )


def _accumulate_tool_calls(accumulator: dict[int, dict[str, str]], value: Any) -> None:
    if not isinstance(value, list):
        return
    for entry in value:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        slot = accumulator.setdefault(
            index if isinstance(index, int) else 0,
            {"id": "", "name": "", "arguments": ""},
        )
        call_id = entry.get("id")
        if isinstance(call_id, str) and call_id:
            slot["id"] = call_id
        function = entry.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                slot["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                slot["arguments"] += arguments


def _snapshot_tool_calls(
    accumulator: Mapping[int, Mapping[str, str]],
) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index in sorted(accumulator):
        slot = accumulator[index]
        name = slot.get("name", "")
        call_id = slot.get("id", "")
        if not name and not call_id:
            continue
        raw = slot.get("arguments") or ""
        arguments, _ = _decode_arguments(raw)
        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                arguments_raw=raw if _is_json_object(raw) else None,
            )
        )
    return calls


def _parse_tool_calls(value: Any) -> list[ToolCall]:
    if not isinstance(value, list):
        return []
    calls: list[ToolCall] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        raw_function = entry.get("function")
        function: dict[str, Any] = (
            raw_function if isinstance(raw_function, dict) else {}
        )
        raw = function.get("arguments")
        arguments: dict[str, Any] = {}
        arguments_raw: str | None = None
        if isinstance(raw, str):
            arguments, arguments_raw = _decode_arguments(raw)
        elif isinstance(raw, dict):
            arguments, arguments_raw = dict(raw), json.dumps(raw)
        elif raw is not None:
            # A number or list here is malformed. Keep its JSON text so
            # execute_tool_calls reports "invalid JSON arguments" rather than
            # running the tool with empty arguments.
            arguments_raw = json.dumps(raw)
        calls.append(
            ToolCall(
                id=_opt_str(entry.get("id")) or "",
                name=_opt_str(function.get("name")) or "",
                arguments=arguments,
                arguments_raw=arguments_raw,
            )
        )
    return calls


def _decode_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """Decode a tool call's ``arguments`` JSON text.

    The raw text is preserved even when it does not parse, so
    :func:`litelm.execute_tool_calls` can tell the model exactly what arrived.
    """
    if raw is None:
        return {}, None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}, raw
    if isinstance(decoded, dict):
        return decoded, raw
    return {}, raw


def _is_json_object(raw: str) -> bool:
    if not raw:
        return False
    try:
        return isinstance(json.loads(raw), dict)
    except json.JSONDecodeError:
        return False


def _parse_usage(value: Any) -> Usage | None:
    if not isinstance(value, dict):
        return None
    prompt = _first_int(value.get("prompt_tokens"), value.get("input_tokens"))
    completion = _first_int(value.get("completion_tokens"), value.get("output_tokens"))
    total = _first_int(value.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _text_of(value: Any) -> str | None:
    """Some providers answer with a content-block list instead of a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            block["text"]
            for block in value
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return None


def _reasoning_of(message: Mapping[str, Any]) -> str | None:
    for key in ("reasoning", "reasoning_content", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value
    return None


TOOL_CHOICES = ("auto", "none", "required")


def _tool_choice(value: ToolChoice) -> Any:
    """Validate ``tool_choice`` the way the Racket port does, so a typo fails
    here with a clear message instead of as an opaque provider 400."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value in TOOL_CHOICES:
        return value
    raise LitelmError(
        f"tool_choice must be 'auto', 'none', 'required', a dict, or None: {value!r}"
    )


def _embedding_inputs(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        inputs = list(value)
        if inputs and all(isinstance(item, str) for item in inputs):
            return inputs
    raise LitelmError(f"embedding input must be a string or list of strings: {value!r}")


def _vector_of(item: Any) -> list[float]:
    vector = item.get("embedding") if isinstance(item, dict) else None
    if not isinstance(vector, list):
        return []
    try:
        # Every element must convert; dropping one would silently return a
        # shorter vector that no longer lines up with the model's dimensions.
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise LitelmError(f"Malformed embedding vector: {_short(vector)}") from exc


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
    return None


def _short(value: Any, limit: int = 300) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."
