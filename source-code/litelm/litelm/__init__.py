"""litelm -- a uniform interface to every LLM provider used in this book.

Models are addressed as ``"provider/model-name"`` strings and one code path
serves them all, so switching between a local Ollama model and a cloud API is a
one-word edit::

    import litelm

    litelm.ask("ollama/llama3.2:3b", "What is 2+2?")
    litelm.ask("openai/gpt-5.4-nano", "What is 2+2?")
    litelm.ask("gemini/gemini-3-flash-preview", "What is 2+2?")
    litelm.ask("fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash", "2+2?")
    litelm.ask("nvidia/meta/llama-3.1-8b-instruct", "What is 2+2?")

This is a Python port of the Common Lisp ``litelm`` library, with the same
interface as the Racket ``llmapis`` port: routing, message handling, streaming,
tools as ordinary functions, embeddings, and an error hierarchy mirroring the
providers' HTTP failures. ``completion``, ``ask``, ``responses``,
``chat_with_tools``, ``embedding`` and ``generate_image`` reach the network;
everything else is pure.

The library needs no third-party packages -- it speaks HTTP with ``urllib`` and
JSON with ``json``, exactly as the Common Lisp version speaks HTTP with
``dexador`` and its own JSON codec.
"""

from __future__ import annotations

from .core import (
    WEB_SEARCH,
    ToolChoice,
    ask,
    build_chat_payload,
    chat_with_tools,
    completion,
    embedding,
    generate_image,
    parse_chat_response,
    parse_responses_body,
    responses,
)
from .errors import (
    ApiError,
    AuthenticationError,
    ContextWindowExceededError,
    LitelmError,
    NotFoundError,
    RateLimitError,
)
from .messages import (
    Message,
    Messages,
    assistant_message,
    normalize_messages,
    tool_message,
)
from .providers import (
    Provider,
    bearer_headers,
    define_provider,
    find_provider,
    parse_model,
    provider_api_key,
    provider_native_url,
    provider_url,
    providers,
)
from .tools import (
    Handler,
    Tool,
    ToolParam,
    execute_tool_calls,
    make_tool,
    normalize_tools,
    param,
    tool_schemas,
)
from .transport import DEFAULT_TIMEOUT
from .types import GeneratedImage, Response, StreamChunk, ToolCall, ToolResult, Usage

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TIMEOUT",
    "WEB_SEARCH",
    "ApiError",
    "AuthenticationError",
    "ContextWindowExceededError",
    "GeneratedImage",
    "Handler",
    "LitelmError",
    "Message",
    "Messages",
    "NotFoundError",
    "Provider",
    "RateLimitError",
    "Response",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolParam",
    "ToolResult",
    "Usage",
    "ask",
    "assistant_message",
    "bearer_headers",
    "build_chat_payload",
    "chat_with_tools",
    "completion",
    "define_provider",
    "embedding",
    "execute_tool_calls",
    "find_provider",
    "generate_image",
    "make_tool",
    "normalize_messages",
    "normalize_tools",
    "param",
    "parse_chat_response",
    "parse_model",
    "parse_responses_body",
    "provider_api_key",
    "provider_native_url",
    "provider_url",
    "providers",
    "responses",
    "tool_message",
    "tool_schemas",
]
