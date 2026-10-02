# litelm — one interface for every LLM in this book

A small, dependency-free Python library that puts a **single interface** on the
LLM providers this book uses: the local Ollama models in
[`../llm_local_models`](../llm_local_models) and the cloud APIs in
[`../llm_public_apis`](../llm_public_apis).

Models are addressed as `"provider/model-name"` strings, so switching from a
local model to a cloud API is a one-word edit:

```python
import litelm

litelm.ask("ollama/llama3.2:3b", "What is 2+2?")  # local, no key
litelm.ask("openai/gpt-5.4-nano", "What is 2+2?")  # OpenAI
litelm.ask("gemini/gemini-3-flash-preview", "What is 2+2?")  # Google Gemini
litelm.ask("nvidia/meta/llama-3.1-8b-instruct", "What is 2+2?")  # NVIDIA NIM
litelm.ask("fireworks-ai/accounts/fireworks/models/deepseek-v4-flash", "2+2?")
```

This is the Python member of a family. The Common Lisp original is
`loving-common-lisp/src/litelm` and the Racket port is
`Racket-AI-book/source-code/llmapis`; all three share the same routing scheme,
the same entry-point names, and the same error hierarchy. Only the surface
syntax differs.

## No dependencies

The library imports nothing outside the standard library — `urllib` for HTTP and
`json` for encoding — exactly as the Common Lisp version uses `dexador` plus its
own JSON codec. `uv sync` is only needed for the *development* tools (ruff,
pyrefly, pytest).

## Model routing

Every provider registered by default speaks the OpenAI-compatible
`/chat/completions` protocol, so one code path serves them all and the prefix of
the model string selects the endpoint:

| Provider | Prefix | API key env vars | Base URL |
|---|---|---|---|
| OpenAI | `openai/` | `OPENAI_API_KEY`, `OPENAI_KEY` | `https://api.openai.com/v1` |
| Google Gemini | `gemini/` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `.../v1beta/openai` |
| Fireworks AI | `fireworks-ai/` | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` |
| NVIDIA NIM | `nvidia/` | `NVIDIA_API_KEY` | `https://integrate.api.nvidia.com/v1` |
| Ollama (local) | `ollama/` | — | `http://localhost:11434/v1` |

Model names may themselves contain slashes. Only the **first** slash separates
the provider from the model, so `nvidia/meta/llama-3.1-8b-instruct` and
`fireworks-ai/accounts/fireworks/models/deepseek-v4-flash` survive intact.

The Common Lisp and Racket versions of litelm also ship `deepseek` and
`mistral`. This port deliberately does not, because the two chapters this
directory backs never call them — they are one `define_provider` call away.

Any other OpenAI-compatible service can be registered at runtime:

```python
litelm.define_provider(
    "groq", "https://api.groq.com/openai/v1", env_keys=["GROQ_API_KEY"]
)
litelm.ask("groq/llama-3.1-8b-instant", "Say hi.")

litelm.define_provider(
    "deepseek", "https://api.deepseek.com/v1", env_keys=["DEEPSEEK_API_KEY"]
)
litelm.ask("deepseek/deepseek-chat", "Say hi.")
```

Per call, `api_key=` and `api_base=` override both the environment variable and
the registered URL.

## Messages

Messages are plain dicts in the order the model sees them — the same shape the
OpenAI SDK uses, so there is nothing new to learn:

```python
messages = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "sunny and 22C"},
]
```

A bare string is shorthand for a single user message, and a lone dict is
accepted too. Malformed input — an unknown role, a `tool` message without its
`tool_call_id`, an assistant message with neither content nor tool calls — raises
`LitelmError` with a message naming the problem, instead of coming back as an
opaque provider 400.

### Images

Attach local files, URLs, or raw bytes with the `images` key on a user message.
litelm reads the file, sniffs its MIME type, base64-encodes it, and emits the
`image_url` content parts the wire expects:

```python
response = litelm.completion(
    "ollama/qwen3.5:0.8b",
    [
        {
            "role": "user",
            "content": "Describe this image in detail",
            "images": ["ticket.png"],
        },
    ],
)
```

## Tools are Python functions

A tool is an ordinary callable taking one `dict` of arguments:

```python
def get_weather(args: dict) -> str:
    return f"sunny and 22C in {args['location']}"


weather = litelm.make_tool(
    "get_weather",
    "Get the current weather for a location",
    [("location", "string", "City name, e.g. Paris")],
    get_weather,
)
```

`parameters` entries are `(name, type, description)` tuples, required by default;
use `litelm.param(...)` when a parameter is optional or has an `enum`:

```python
litelm.param(
    "units",
    "string",
    "celsius or fahrenheit",
    required=False,
    enum=("celsius", "fahrenheit"),
)
```

**Tool calls are returned, not executed.** `completion` hands you the calls and
`execute_tool_calls` runs them when *you* decide to:

```python
response = litelm.completion(model, messages, tools=[weather])
for call in response.tool_calls:
    print(call.name, dict(call.arguments))
```

Or let `chat_with_tools` run the whole request / execute / reply loop:

```python
answer = litelm.chat_with_tools(model, "Weather in Paris?", [weather])
```

Failures — an unknown tool, malformed JSON arguments, a missing required
argument, an exception inside the handler — become `"Error: ..."` result strings
for the model to read. They are never raised into your program.

For manual control of the loop, `assistant_message(response)` and
`tool_message(result)` build the two continuation messages.

## Entry points

| Call | Returns |
|---|---|
| `completion(model, messages, **options)` | `Response` |
| `completion(model, messages, stream=True)` | iterator of `StreamChunk` |
| `ask(model, prompt)` | the reply text |
| `embedding(model, input)` | `list[list[float]]` |
| `chat_with_tools(model, messages, tools)` | the final `Response` |

Shared keyword options: `tools`, `tool_choice` (`"auto"` / `"none"` /
`"required"` / `None`), `temperature`, `max_tokens`, `top_p`, `system`,
`provider`, `api_key`, `api_base`, `extra_headers`, `extra`, `timeout`.

`Response` carries `content`, `tool_calls`, `finish_reason`, `model`, `usage`
(`Usage(prompt_tokens, completion_tokens, total_tokens)`), `reasoning`, and
`raw` (the decoded JSON body). `ToolCall` carries `id`, `name`, `arguments`
(a dict), and `arguments_raw` (the JSON text as it arrived).

`timeout` defaults to 300 seconds (`litelm.DEFAULT_TIMEOUT`) — higher than the
120 seconds the Common Lisp and Racket ports use, because the local reasoning
models this book runs can spend minutes inside one non-streamed answer. Raise it
further for a big quantised model on a long context, or lower it to fail fast:

```python
litelm.completion("ollama/nimble:9b", long_conversation, timeout=900)
litelm.completion("openai/gpt-5.4-nano", "hi", timeout=30)
```

A timeout raises `LitelmError` like any other transport failure, so it is not a
special case in error handling.

### Streaming

```python
for chunk in litelm.completion("ollama/llama3.2:3b", "Count to five.", stream=True):
    print(chunk.text, end="", flush=True)
```

`chunk.text` is the incremental delta; `chunk.finish_reason` lands on the last
chunk. Providers send tool calls in fragments, so litelm accumulates them and
every chunk carries the complete set so far in `chunk.tool_calls`.

### Embeddings

`embedding` returns one `list[float]` per input, in input order. Two malformed
responses raise `LitelmError` rather than returning something quietly wrong: a
body with no `data` array, and a vector containing a non-numeric element (the
Common Lisp and Racket versions return an empty list and pass the vector
through respectively — this port is stricter on purpose, since a short vector
misaligns against the model's dimensions).

### Provider-specific options

Anything a provider supports beyond the common set travels in `extra`, which is
merged into the request body last:

```python
# Fireworks / DeepSeek thinking mode
litelm.completion(
    "fireworks-ai/accounts/fireworks/models/deepseek-v4-flash",
    prompt,
    extra={"thinking": {"type": "enabled"}},
)

# Ollama's vision models print a <think> block unless told not to
litelm.completion("ollama/qwen3.5:0.8b", messages, extra={"think": False})
```

When a provider returns a reasoning trace as a separate field, it is surfaced as
`response.reasoning`. DeepSeek-R1 served by Ollama instead leaves its
`<think>...</think>` block inline in `response.content`; [`examples/example_thinking.py`](examples/example_thinking.py)
shows both cases.

## Errors

HTTP failures map onto an exception hierarchy mirroring the Common Lisp and
Racket versions:

```
LitelmError
└── ApiError                      readers: status, body
    ├── AuthenticationError       401, 403
    ├── RateLimitError            429
    ├── NotFoundError             404
    └── ContextWindowExceededError 400 mentioning "context"
```

```python
try:
    litelm.completion(model, messages)
except litelm.RateLimitError:
    ...  # back off and retry
except litelm.AuthenticationError:
    ...  # bad or missing API key
except litelm.ApiError as error:
    print(error.status, error.body)
```

Two provider quirks are handled for you. OpenAI models that reject
`max_tokens` and demand `max_completion_tokens` are retried once with the
parameter renamed — on the streaming path as well as the ordinary one. And a
provider that reports a failure as an `error` event *inside* a 200 SSE stream
raises `LitelmError` instead of ending the iterator silently.

## Examples

The runnable demos live in [`examples/`](examples) — one script per idea from the
two chapters:

| Script | Shows |
|---|---|
| [`example_text.py`](examples/example_text.py) | Basic text generation; switch providers with `LITELM_MODEL` |
| [`example_streaming.py`](examples/example_streaming.py) | Streaming deltas |
| [`example_conversation.py`](examples/example_conversation.py) | Multi-turn history |
| [`example_tools.py`](examples/example_tools.py) | Tool calling, both the manual loop and `chat_with_tools` |
| [`example_embeddings.py`](examples/example_embeddings.py) | Embeddings and cosine similarity |
| [`example_vision.py`](examples/example_vision.py) | Image understanding |
| [`example_thinking.py`](examples/example_thinking.py) | Reasoning traces, separate and inline |
| [`example_structured.py`](examples/example_structured.py) | JSON output extraction |
| [`example_nvidia.py`](examples/example_nvidia.py) | NVIDIA NIM, mirroring `NVIDIA_client.py` |

Run any of them with `uv run`, or through `make`:

```bash
uv run python examples/example_text.py
LITELM_MODEL=openai/gpt-5.4-nano uv run python examples/example_text.py
make tools
```

Local models need `ollama serve` and a pulled model
(`ollama pull llama3.2:3b`). Cloud models need the matching environment
variable shown in the routing table.

## Installing litelm as a local library

`litelm` is an ordinary installable package (a `[build-system]` plus a `litelm`
package), not just a folder of scripts, so any other project in this book — or
any project on your machine — can depend on it.

`uv sync` already installs it editable into this directory's `.venv`, which is
why `uv run python examples/example_text.py` imports it. For everything else:

```bash
make install      # uv pip install --editable .   -> into this project's .venv
make uninstall    # uv pip uninstall litelm
make build        # uv build                      -> dist/*.whl and *.tar.gz
```

Because the install is **editable**, edits to `litelm/*.py` take effect straight
away in the consuming environment — no reinstall while you follow the book.

To use it from a sibling chapter directory, let uv record the dependency:

```bash
cd ../llm_local_models
uv add --editable ../litelm
uv run python -c "import litelm; print(litelm.ask('ollama/llama3.2:3b', 'hi'))"
```

`uv add` writes the path into that project's `pyproject.toml` and lockfile, so
the dependency survives `uv sync`. Use `uv remove litelm` to undo it, or plain
`uv pip install --editable ../litelm` if you would rather not touch its
`pyproject.toml`.

## How this relates to the two chapter directories

The demos in `llm_local_models` and `llm_public_apis` drive each SDK directly,
which is the point of those chapters: `ollama.chat(...)`, `client.models.
generate_content(...)`, `client.responses.create(...)`. This library is the
after-the-fact generalisation — one call shape, five providers — and the mapping
is mechanical:

| Chapter code | litelm |
|---|---|
| `ollama.chat(model="llama3.2:3b", messages=[...])` | `litelm.completion("ollama/llama3.2:3b", [...])` |
| `client.chat.completions.create(base_url=..., ...)` | `litelm.completion("fireworks-ai/...", ...)` |
| `client.responses.create(model="gpt-5.4-nano", input=...)` | `litelm.ask("openai/gpt-5.4-nano", ...)` |
| `for chunk in ollama.chat(..., stream=True)` | `for chunk in litelm.completion(..., stream=True)` |
| `client.models.generate_content(model=..., contents=...)` | `litelm.completion("gemini/...", ...)` |
| `types.Part.from_bytes(data=..., mime_type=...)` | `{"role": "user", "content": ..., "images": [path]}` |

Features that only one provider has — Gemini's Google Search grounding, OpenAI's
`web_search_preview` tool, the Responses API itself — deliberately stay in the
per-provider chapter scripts.

One consequence worth stating plainly: litelm always speaks
`POST /chat/completions`. For an OpenAI model that is *only* reachable through
the Responses API, `litelm.completion("openai/...")` is the wrong tool — use
[`../llm_public_apis/openai_text.py`](../llm_public_apis/openai_text.py) or
`openai_search.py` directly. The `max_completion_tokens` retry described under
[Errors](#errors) exists because newer OpenAI models reject the older
`max_tokens` spelling on this endpoint.

## File structure

| File | Contents |
|---|---|
| `litelm/__init__.py` | Public interface and re-exports |
| `litelm/providers.py` | Provider registry and `"provider/model"` routing |
| `litelm/messages.py` | Message normalization, images, tool-message construction |
| `litelm/tools.py` | `Tool`/`ToolParam`, JSON schemas, `execute_tool_calls` |
| `litelm/core.py` | `completion`, `ask`, `embedding`, `chat_with_tools`, parsing |
| `litelm/transport.py` | The only module that touches the network (HTTP + SSE) |
| `litelm/types.py` | `Response`, `StreamChunk`, `ToolCall`, `ToolResult`, `Usage` |
| `litelm/errors.py` | The exception hierarchy |
| `examples/` | The nine runnable demos (import-safe, `main()`-guarded) |
| `tests/` | Offline suite (fake transport, loopback HTTP server) plus opt-in live tests |

## Development workflow

```bash
uv sync
just check       # fmt-check + lint + typecheck + test
just fmt         # format all Python files
just lint        # ruff --fix
just typecheck   # pyrefly (strict preset)
just test        # pytest (offline; live tests are deselected)
```

`make check` runs the same four gates without `just`.

## Tests

The offline suite replaces the HTTP layer, so it asserts on the exact JSON bodies
and headers litelm would send — 183 tests, no network, no keys, well under a
second:

```bash
uv run pytest -q                       # offline only (default)
uv run pytest -q -m live               # real providers, skipped when unavailable
OLLAMA_MODEL=nimble:9b uv run pytest -q -m live
```

Live tests are opt-in and skip themselves when the matching API key is unset or
no local Ollama server is reachable.

## License

Copyright (C) 2026 Mark Watson — Apache 2 License
