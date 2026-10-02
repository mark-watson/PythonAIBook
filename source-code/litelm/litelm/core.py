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

    litelm.completion("fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash",
                      "Solve the fox, chicken and grain puzzle.",
                      extra={"thinking": {"type": "enabled"}})
"""

from __future__ import annotations

import base64
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
    provider_native_url,
    provider_url,
)
from .tools import (
    Tool,
    execute_tool_calls,
    tool_schemas,
)
from .types import GeneratedImage, Response, StreamChunk, ToolCall, Usage

__all__ = [
    "WEB_SEARCH",
    "ask",
    "build_chat_payload",
    "chat_with_tools",
    "completion",
    "embedding",
    "generate_image",
    "parse_chat_response",
    "parse_responses_body",
    "responses",
]

#: Tools may be a list or a ``name -> Tool`` mapping.
Tools = Sequence[Tool] | Mapping[str, Tool]
#: ``tool_choice`` values every OpenAI-compatible provider understands.
ToolChoice = str | dict[str, Any] | None

#: The Responses API's built-in web-search tool, for
#: ``responses(..., tools=[litelm.WEB_SEARCH])``. It runs on the provider's
#: side -- litelm never sees the search results, only the answer that cites
#: them. Handed over as a copy, so a caller cannot mutate the shared constant.
WEB_SEARCH: dict[str, str] = {"type": "web_search_preview"}


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


def parse_responses_body(data: Mapping[str, Any], model_name: str) -> Response:
    """Turn a Responses API (``POST /responses``) body into a :class:`Response`.

    The Responses API answers with an ``output`` *array* of typed items rather
    than a single message: ``message`` items hold the text, ``reasoning`` items
    hold a summary trace, and ``function_call`` items hold tool calls. A failure
    can arrive as an ``error`` field, or as ``status: "failed"``.
    """
    if data.get("error"):
        raise LitelmError(f"LLM API error: {data['error']}")
    status = _opt_str(data.get("status"))
    if status == "failed":
        raise LitelmError(f"LLM API response failed: {_short(data)}")

    finish_reason = status or _opt_str(data.get("finish_reason"))
    if status == "incomplete":
        details = data.get("incomplete_details")
        if isinstance(details, dict) and isinstance(details.get("reason"), str):
            finish_reason = details["reason"]

    explicit_text = _opt_str(data.get("output_text"))
    return Response(
        content=explicit_text if explicit_text is not None else _responses_text(data),
        tool_calls=_responses_tool_calls(data),
        finish_reason=finish_reason,
        model=_opt_str(data.get("model")) or model_name,
        usage=_parse_usage(data.get("usage")),
        reasoning=_responses_reasoning(data),
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
# responses (the OpenAI Responses API)
# --------------------------------------------------------------------------


def responses(
    model: str,
    input: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    instructions: str | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
) -> Response:
    """Call the OpenAI **Responses API** and return a :class:`Response`.

    The Responses API is a different protocol from ``/chat/completions``, so it
    gets its own entry point rather than an option on :func:`completion`:

    * ``input`` is a prompt string or a list of ``{"role": ..., "content": ...}``
      items. Use ``instructions`` for the system prompt -- the Responses API has
      no ``system`` role.
    * ``tools`` holds *provider-side* tools such as :data:`WEB_SEARCH`, not
      litelm :class:`~litelm.tools.Tool` functions. The model runs them and the
      final answer comes back in ``content``.
    * ``max_output_tokens`` is the Responses spelling of ``max_tokens``.

    ::

        litelm.responses("openai/gpt-5.4-nano", "What is 2+2?")
        litelm.responses("openai/gpt-5.4-nano", "AI news this week?",
                         tools=[litelm.WEB_SEARCH])
    """
    prov, model_name = parse_model(model, provider)
    key = provider_api_key(prov, api_key)
    headers = bearer_headers(key)
    if extra_headers:
        headers.update(extra_headers)

    payload: dict[str, Any] = {"model": model_name, "input": _responses_input(input)}
    if instructions is not None:
        payload["instructions"] = instructions
    if tools:
        # Copies: the caller keeps ownership of its tool dicts.
        payload["tools"] = [dict(tool) for tool in tools]
    if temperature is not None:
        payload["temperature"] = temperature
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if extra:
        payload.update(extra)

    data = transport.post_json(
        provider_url(prov, "/responses", api_base), headers, payload, timeout=timeout
    )
    return parse_responses_body(data, model_name)


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
# image generation (the provider's own API)
# --------------------------------------------------------------------------


def generate_image(
    model: str,
    prompt: str,
    *,
    number_of_images: int = 1,
    aspect_ratio: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    timeout: float = transport.DEFAULT_TIMEOUT,
) -> list[GeneratedImage]:
    """Generate images from ``prompt``; returns one :class:`GeneratedImage` each.

    Text-to-image has no OpenAI-compatible equivalent, so this reaches the
    provider's own API through ``provider.native_url`` -- today Gemini's Imagen
    (``gemini/imagen-4.0-fast-generate-001``). A provider without a native URL
    raises :class:`LitelmError` naming the problem.

    ``extra`` is merged into the request's ``parameters`` object for
    model-specific knobs (``negativePrompt``, ``personGeneration``, ...).
    """
    prov, model_name = parse_model(model, provider)
    base = provider_native_url(prov, api_base)
    key = provider_api_key(prov, api_key)
    headers = {"Content-Type": "application/json"}
    if key:
        # The Gemini API authenticates its native endpoints with this header.
        headers["x-goog-api-key"] = key
    if extra_headers:
        headers.update(extra_headers)

    parameters: dict[str, Any] = {"sampleCount": number_of_images}
    if aspect_ratio is not None:
        parameters["aspectRatio"] = aspect_ratio
    if extra:
        parameters.update(extra)
    payload: dict[str, Any] = {
        "instances": [{"prompt": prompt}],
        "parameters": parameters,
    }

    data = transport.post_json(
        f"{base}/models/{model_name}:predict", headers, payload, timeout=timeout
    )
    return _images_of(data)


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
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=_cached_tokens(value),
    )


def _cached_tokens(value: Mapping[str, Any]) -> int | None:
    """The provider's count of prompt tokens served from its cache.

    Every provider spells this differently, and a server that reports no cache
    numbers at all must not look like a cache miss -- so this returns ``None``
    rather than zero when the field is absent.
    """
    details = value.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = _first_int(details.get("cached_tokens"))
        if cached is not None:
            return cached
    return _first_int(
        value.get("cache_read_input_tokens"),  # Anthropic
        value.get("prompt_cache_hit_tokens"),  # DeepSeek
        value.get("cached_tokens"),
    )


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


def _responses_input(
    value: str | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> str | list[dict[str, Any]]:
    """Normalize ``responses`` input to what the wire wants: text or items."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence):
        items: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise LitelmError(f"responses input entries must be dicts: {item!r}")
            items.append(dict(item))
        return items
    raise LitelmError(
        f"responses input must be a string or a list of message dicts: {value!r}"
    )


def _responses_items(output: Any) -> list[Mapping[str, Any]]:
    if not isinstance(output, list):
        return []
    return [item for item in output if isinstance(item, Mapping)]


def _responses_text(data: Mapping[str, Any]) -> str | None:
    """Concatenate the ``output_text`` parts of every ``message`` item."""
    parts: list[str] = []
    for item in _responses_items(data.get("output")):
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, Mapping)
                and block.get("type") == "output_text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
    return "".join(parts) or None


def _responses_reasoning(data: Mapping[str, Any]) -> str | None:
    """Join the ``summary_text`` parts of every ``reasoning`` item."""
    parts: list[str] = []
    for item in _responses_items(data.get("output")):
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if not isinstance(summary, list):
            continue
        for block in summary:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "".join(parts) or None


def _responses_tool_calls(data: Mapping[str, Any]) -> list[ToolCall]:
    """Turn ``function_call`` output items into :class:`ToolCall` values."""
    calls: list[ToolCall] = []
    for item in _responses_items(data.get("output")):
        if item.get("type") != "function_call":
            continue
        raw = item.get("arguments")
        arguments: dict[str, Any] = {}
        arguments_raw: str | None = None
        if isinstance(raw, str):
            arguments, arguments_raw = _decode_arguments(raw)
        elif isinstance(raw, Mapping):
            arguments, arguments_raw = dict(raw), json.dumps(raw)
        calls.append(
            ToolCall(
                # The Responses API names it call_id; older bodies use id.
                id=_opt_str(item.get("call_id")) or _opt_str(item.get("id")) or "",
                name=_opt_str(item.get("name")) or "",
                arguments=arguments,
                arguments_raw=arguments_raw,
            )
        )
    return calls


def _images_of(data: Mapping[str, Any]) -> list[GeneratedImage]:
    """Decode the base64 images of an Imagen-style ``predictions`` array."""
    if data.get("error"):
        raise LitelmError(f"LLM API error: {data['error']}")
    predictions = data.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise LitelmError(f"Image response has no predictions: {_short(data)}")
    images: list[GeneratedImage] = []
    for item in predictions:
        if not isinstance(item, Mapping):
            continue
        encoded = item.get("bytesBase64Encoded")
        mime = item.get("mimeType")
        nested = item.get("image")
        if not isinstance(encoded, str) and isinstance(nested, Mapping):
            raw = nested.get("bytesBase64Encoded") or nested.get("imageBytes")
            encoded = raw if isinstance(raw, str) else None
            if not isinstance(mime, str):
                mime = nested.get("mimeType")
        if not isinstance(encoded, str):
            continue
        try:
            decoded = base64.b64decode(encoded)
        except ValueError as exc:  # binascii.Error is a ValueError
            raise LitelmError("Image response carried malformed base64 data") from exc
        images.append(
            GeneratedImage(
                data=decoded,
                mime_type=mime if isinstance(mime, str) and mime else "image/png",
            )
        )
    if not images:
        raise LitelmError(f"Image response had no decodable images: {_short(data)}")
    return images


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
