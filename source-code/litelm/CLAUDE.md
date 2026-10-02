# litelm

A dependency-free Python library — and the eleven runnable demos around it —
that puts **one interface on every LLM provider this book uses**: the local
Ollama models from `llm_local_models` and the cloud APIs from
`llm_public_apis`. Every other LLM example in `source-code/` depends on it.

Models are addressed as `"provider/model-name"` strings, messages are plain
dicts, tools are ordinary Python functions, and the whole library talks HTTP
with `urllib` — no SDKs, no runtime dependencies.

```python
import litelm

litelm.ask("ollama/llama3.2:3b", "What is 2+2?")  # local, no key
litelm.ask("openai/gpt-5.4-nano", "What is 2+2?")
litelm.ask("gemini/gemini-3-flash-preview", "What is 2+2?")
litelm.ask("nvidia/meta/llama-3.1-8b-instruct", "What is 2+2?")
litelm.ask("fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash", "2+2?")
```

It is the Python member of a family: the Common Lisp original lives in
`loving-common-lisp/src/litelm`, the Racket port in
`Racket-AI-book/source-code/llmapis`. All three share the routing scheme, the
entry-point names (`completion` / `ask` / `embedding` / `chat_with_tools`), the
tool-call contract, and the error hierarchy. The Python port adds two entry
points the other two do not have -- `responses` for the OpenAI Responses API and
`generate_image` for Imagen -- so this book's examples can reach those
capabilities through the same library.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

Local models need `ollama serve` running and a model pulled
(`ollama pull llama3.2:3b`). Cloud models need the environment variable from the
routing table in `README.md`.

## Layout

```
litelm/
├── litelm/
│   ├── __init__.py            # public interface (__all__ is the contract)
│   ├── providers.py           # registry, "provider/model" routing, native URLs
│   ├── messages.py            # normalization, images, assistant/tool messages
│   ├── tools.py               # Tool/ToolParam, JSON schemas, execute_tool_calls
│   ├── core.py                # completion/ask/responses/embedding/generate_image
│   ├── transport.py           # the ONLY module that touches the network
│   ├── types.py               # Response, StreamChunk, ToolCall, ToolResult, Usage
│   │                          #   GeneratedImage
│   └── errors.py              # LitelmError -> ApiError -> 4 subclasses
├── examples/                  # runnable demos, import-safe (main()-guarded)
│   ├── __init__.py
│   ├── example_text.py        # basic generation, provider chosen by LITELM_MODEL
│   ├── example_streaming.py
│   ├── example_conversation.py
│   ├── example_tools.py
│   ├── example_embeddings.py
│   ├── example_vision.py
│   ├── example_thinking.py
│   ├── example_structured.py
│   ├── example_search.py      # Responses API + the built-in web-search tool
│   ├── example_image.py       # Imagen text-to-image
│   └── example_nvidia.py
├── tests/
│   ├── conftest.py            # FakeTransport: offline HTTP double
│   ├── test_providers.py      # routing, registry, API key resolution
│   ├── test_messages.py       # normalization, images, tool messages
│   ├── test_tools.py          # schemas, execution, the agentic loop
│   ├── test_completion.py     # payload building, response parsing, fallbacks
│   ├── test_responses.py      # Responses API bodies and output-item parsing
│   ├── test_images.py         # native predict requests and decoded images
│   ├── test_streaming.py      # SSE chunk folding, tool-call fragments
│   ├── test_transport.py      # status mapping, JSON decoding, SSE framing
│   ├── test_http_server.py    # end-to-end over a real loopback socket
│   ├── test_examples.py       # examples import and expose main()
│   └── test_live.py           # opt-in, skipped without keys/Ollama
├── pyproject.toml             # also declares the build backend (installable)
├── pyrefly.toml
├── justfile
└── Makefile                   # includes `make install` / `make uninstall`
```

## Workflow rules

`just check` (equivalently `make check`) must pass before finishing any change:

```
ruff format --check .   ruff check .   pyrefly check   pytest -q
```

- `pyrefly.toml` is `preset = "strict"` with `python-version = "3.14"`, matching
  the sibling chapter directories. `project-includes` lists the three
  directories (`litelm`, `examples`, `tests`), so new files inside them are
  picked up automatically — but a **new top-level directory** must be added
  there or it will not be typechecked.
- The package carries **zero runtime dependencies**. Do not add one; if a
  provider is awkward, put the difference behind a new provider kind or an
  `extra=` passthrough, not behind an SDK import.
- `litelm/transport.py` is the only module allowed to import `urllib` or touch
  the network. Everything else must stay testable without a server.
- `litelm/__init__.py::__all__` is the public contract and is kept sorted
  (ruff's RUF022 enforces the order). Keep `README.md` in step with it.

## Testing notes

- The default run is offline and fast: `tests/conftest.py` installs a
  `FakeTransport` in place of `litelm.transport`, so tests assert on the exact
  JSON bodies, headers, URLs and timeouts litelm would have sent. That is where
  provider bugs live.
- An autouse fixture clears provider API-key environment variables so key
  resolution is deterministic on any machine. Tests marked `live` are exempt.
- `tests/test_live.py` is opt-in: `addopts = "-m 'not live'"` deselects it, and
  each test additionally `skipif`s when its key or server is missing. Run them
  with `uv run pytest -q -m live` (the CLI `-m` overrides `addopts`).
- Live tests must not assert on model *wording*. Small local models are
  inconsistent; assert that the protocol round-tripped, not what it said.
- `tests/test_http_server.py` starts a throwaway OpenAI-compatible server on
  a loopback port, so the real `urllib` path — headers, JSON round trip, SSE
  framing, status mapping, the tool loop — is exercised end to end without a
  third-party provider.
- The suite is mutation-checked: deleting the error branch in
  `parse_chat_response`, ignoring the `Mapping` tool registry, un-sorting
  streamed tool-call indices, or dropping raw tool arguments must all fail some
  test. Re-run that check when adding tests around parsing.
- Examples live in the `examples/` package and guard all work behind `main()`
  with an `if __name__ == "__main__":` entry, so `tests/test_examples.py` can
  import them (`examples.example_text`) and exercise their pure helpers. Keep
  that property: no network calls at import time.
- Because the examples now sit one level down, `import litelm` inside them
  resolves through the installed package, not through `sys.path[0]`. That makes
  `uv sync` / `make install` load-bearing: if `uv run python
  examples/example_text.py` reports `ModuleNotFoundError: litelm`, the project
  is not installed in the environment.

## Typing discipline

- Annotate every function, including tests and fixtures.
- Prefer narrowing (`isinstance`) and `typing.cast` over `# type: ignore`; an
  unused ignore is itself a diagnostic.
- `completion` is overloaded on `stream`: `stream=True` returns
  `Iterator[StreamChunk]`, otherwise a `Response`. Keep the overloads and the
  implementation signature in step.
- pyrefly flags empty container literals under strict mode — annotate them.
