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
litelm.ask("fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash", "2+2?")
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

Gemini is the one provider that also carries a **native** URL
(`https://generativelanguage.googleapis.com/v1beta`), because Imagen image
generation has no OpenAI-compatible endpoint. `define_provider(..., native_url=...)`
registers one for any other service that needs it.

Model names may themselves contain slashes. Only the **first** slash separates
the provider from the model, so `nvidia/meta/llama-3.1-8b-instruct` and
`fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash` survive intact.

The Common Lisp and Racket versions of litelm also ship `deepseek` and
`mistral`. This port deliberately does not, because the chapters this
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
| `responses(model, input, tools=...)` | `Response` (the OpenAI Responses API) |
| `embedding(model, input)` | `list[list[float]]` |
| `generate_image(model, prompt)` | `list[GeneratedImage]` |
| `chat_with_tools(model, messages, tools)` | the final `Response` |

Shared keyword options: `tools`, `tool_choice` (`"auto"` / `"none"` /
`"required"` / `None`), `temperature`, `max_tokens`, `top_p`, `system`,
`provider`, `api_key`, `api_base`, `extra_headers`, `extra`, `timeout`.
`responses` swaps `messages` for `input` + `instructions` and `max_tokens` for
`max_output_tokens`, and takes provider-side tools such as `litelm.WEB_SEARCH`
rather than Python functions; `generate_image` returns `GeneratedImage` values
carrying `data`, `mime_type`, `suffix`, and a `save(path)` helper.

`Response` carries `content`, `tool_calls`, `finish_reason`, `model`, `usage`
(`Usage(prompt_tokens, completion_tokens, total_tokens, cached_tokens)` —
`cached_tokens` is the cache-hit count Ollama, DeepSeek and Anthropic report
under their various spellings), `reasoning`, and
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

### The Responses API and provider-side tools

`/chat/completions` is not the only protocol in play: OpenAI's newer models are
also served by the **Responses API**, whose answers arrive as an `output` array
of typed items and whose built-in tools run on the provider's side. `responses`
speaks it:

```python
response = litelm.responses("openai/gpt-5.4-nano", "What is 2+2?")
print(response.content)

# web_search_preview runs at OpenAI; litelm only sees the answer
response = litelm.responses(
    "openai/gpt-5.4-nano",
    "What were the major AI announcements at Google I/O 2025?",
    tools=[litelm.WEB_SEARCH],
)
print(response.content)
```

`input` is a prompt string or a list of `{"role": ..., "content": ...}` items,
`instructions` is the system prompt (the Responses API has no `system` role),
and `finish_reason` is the response `status` (`"completed"`, or the
`incomplete_details.reason`). Reasoning items surface as `response.reasoning`,
and `function_call` items as `response.tool_calls`.

### Image generation

Text-to-image has no OpenAI-compatible endpoint, so `generate_image` reaches the
provider's **native** API — today Gemini's Imagen, registered with a
`native_url`:

```python
for image in litelm.generate_image(
    "gemini/imagen-4.0-fast-generate-001",
    "a serene mountain landscape at sunset, oil painting style",
):
    image.save(f"landscape.{image.suffix}")
```

Each result is a `GeneratedImage(data, mime_type)` with `suffix` and
`save(path)`. `number_of_images`, `aspect_ratio` and `extra` (merged into the
request's `parameters`) cover the model knobs; a provider with no native URL
raises `LitelmError` naming the problem.

### Provider-specific options

Anything a provider supports beyond the common set travels in `extra`, which is
merged into the request body last:

```python
# Fireworks / DeepSeek thinking mode
litelm.completion(
    "fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash",
    prompt,
    extra={"thinking": {"type": "enabled"}},
)

# Ollama's vision models print a <think> block unless told not to
litelm.completion("ollama/qwen3.5:0.8b", messages, extra={"think": False})

# Gemini's OpenAI-compatible layer reads OpenAI's reasoning_effort
litelm.completion(
    "gemini/gemini-3-flash-preview", prompt, extra={"reasoning_effort": "low"}
)
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

The runnable demos live in [`examples/`](examples) — one script per idea the
book's LLM examples use:

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
| [`example_search.py`](examples/example_search.py) | The Responses API and the built-in web-search tool |
| [`example_image.py`](examples/example_image.py) | Imagen text-to-image generation |
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

## How this relates to the chapter directories

**Every LLM example in `source-code/` now goes through litelm.** The chapter
directories — `llm_local_models`, `llm_public_apis`, `text-adventure-game`,
`semantic_web_LLM`, `openknowledge_format`, `deep_learning_image_generation` and
`NVIDIA_Object_Oriented_Agents` — declare it as an editable path dependency, so
one call shape reaches five providers and switching between a local model and a
cloud API is a one-word edit to the model string.

The mapping from the SDK spellings the chapters used to describe is mechanical:

| Chapter code | litelm |
|---|---|
| `ollama.chat(model="llama3.2:3b", messages=[...])` | `litelm.completion("ollama/llama3.2:3b", [...])` |
| `client.chat.completions.create(base_url=..., ...)` | `litelm.completion("fireworks-ai/...", ...)` |
| `client.responses.create(model="gpt-5.4-nano", input=...)` | `litelm.responses("openai/gpt-5.4-nano", ...)` |
| `client.responses.create(..., tools=[{"type": "web_search_preview"}])` | `litelm.responses(..., tools=[litelm.WEB_SEARCH])` |
| `for chunk in ollama.chat(..., stream=True)` | `for chunk in litelm.completion(..., stream=True)` |
| `client.models.generate_content(model=..., contents=...)` | `litelm.completion("gemini/...", ...)` |
| `types.Part.from_bytes(data=..., mime_type=...)` | `{"role": "user", "content": ..., "images": [path]}` |
| `client.models.generate_images(model="imagen-...", prompt=...)` | `litelm.generate_image("gemini/imagen-...", prompt)` |

Two things deliberately have no litelm equivalent, and the chapter scripts keep
them: Gemini's Google Search grounding for `generate_content`, and the
third-party `nooa` agent framework in `NVIDIA_Object_Oriented_Agents`, which
drives its own litellm client and only uses litelm for the endpoint constants
and its direct calls.

`completion` and `responses` are separate entry points because they are separate
protocols: a model reachable *only* through the Responses API needs
`litelm.responses`, not `litelm.completion`. The `max_completion_tokens` retry
described under [Errors](#errors) exists because newer OpenAI models reject the
older `max_tokens` spelling on `/chat/completions`.

## File structure

| File | Contents |
|---|---|
| `litelm/__init__.py` | Public interface and re-exports |
| `litelm/providers.py` | Provider registry, `"provider/model"` routing, native URLs |
| `litelm/messages.py` | Message normalization, images, tool-message construction |
| `litelm/tools.py` | `Tool`/`ToolParam`, JSON schemas, `execute_tool_calls` |
| `litelm/core.py` | `completion`, `ask`, `responses`, `embedding`, `generate_image`, `chat_with_tools`, parsing |
| `litelm/transport.py` | The only module that touches the network (HTTP + SSE) |
| `litelm/types.py` | `Response`, `StreamChunk`, `ToolCall`, `ToolResult`, `Usage`, `GeneratedImage` |
| `litelm/errors.py` | The exception hierarchy |
| `examples/` | The eleven runnable demos (import-safe, `main()`-guarded) |
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
and headers litelm would send — 227 tests, no network, no keys, well under a
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
