# llm_local_models

Seven short demos of running LLMs locally via [Ollama](https://ollama.com/) — plain text, streaming, prompt caching, conversation memory, chain-of-thought reasoning, OpenAI-SDK compatibility, and vision. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

Ollama must be running locally (`ollama serve`) with the required models pulled before you run any of the scripts:

```bash
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b
ollama pull qwen3.5:0.8b   # or a vision-capable model for the image demo
```

## Layout

```
llm_local_models/
├── ollama_text.py                 # simplest text-in / text-out
├── ollama_streaming.py            # streamed chat response
├── ollama_caching.py              # prompt-cache benchmark
├── ollama_memory.py               # multi-turn convo with in-process history
├── ollama_reasoning.py            # DeepSeek-R1 <think>...</think> extraction
├── ollama_openai_compat.py        # driving Ollama via the openai SDK
├── image_to_text_description.py   # vision model describing ticket.png
├── tests/
│   ├── conftest.py                # sys.path shim
│   └── test_smoke.py              # AST-parse (no Ollama server needed)
├── pyproject.toml
├── pyrefly.toml
├── justfile
├── Makefile
└── .claude/
    ├── settings.json
    └── hooks/{py-check.sh,py-stop.sh}
```

## Workflow rules

After any Python edit, `.claude/hooks/py-check.sh` runs automatically — it formats the file with ruff, applies safe autofixes, then typechecks it with pyrefly. Fix any reported errors before moving on.

When Claude finishes a turn, `.claude/hooks/py-stop.sh` runs the full gate (`ruff format --check`, `ruff check`, `pyrefly check`, `pytest`). If it fails, Claude must fix the errors before the session ends.

## Tools

| Command | What it does |
|---------|-------------|
| `just check` | Full gate: fmt-check + lint + typecheck + test |
| `just fmt` / `just lint` / `just typecheck` / `just test` | Individual steps |
| `make text` | `ollama_text.py` |
| `make streaming` | `ollama_streaming.py` |
| `make caching` | `ollama_caching.py` |
| `make memory` | `ollama_memory.py` |
| `make reasoning` | `ollama_reasoning.py` |
| `make openai-compat` | `ollama_openai_compat.py` |
| `make image` | `image_to_text_description.py` |

## Testing notes

- None of the scripts guard their work behind `if __name__ == "__main__":` — they all call `ollama.chat(...)` at module top-level. Importing them from a test would therefore contact the Ollama server (or fail hard if it's not running).
- `test_smoke.py` sidesteps that by `ast.parse`-ing each script instead of executing it. This catches syntax errors and mid-file typos without needing Ollama, but it does not check that model calls actually work — for that, run each demo manually against a live server.
- If you refactor a script into a `main()` function with an `if __name__ == "__main__":` guard, feel free to promote its entry in `test_smoke.py` from a parse test to a real import test.

## Typing discipline

- `pyrefly.toml` is on `preset = "strict"` with `python-version = "3.14"`.
- `ollama` and `openai` SDKs have their own stubs; expect optional-return narrowing to be the main lint category.
