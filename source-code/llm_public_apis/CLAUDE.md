# llm_public_apis

Thirteen short demos of calling cloud LLM APIs: **Google Gemini** (6), **Fireworks** via the OpenAI-compatible endpoint (5), and **OpenAI** direct (2). Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

Set the appropriate environment variables before running any script:

```bash
export GOOGLE_API_KEY="..."      # for gemini_*
export FIREWORKS_API_KEY="..."   # for fireworks_*
export OPENAI_API_KEY="..."      # for openai_*
```

## Layout

```
llm_public_apis/
├── gemini_text.py                # simple prompt/response
├── gemini_conversation.py        # multi-turn conversation
├── gemini_image.py               # multimodal image + prompt
├── gemini_structured.py          # JSON output
├── gemini_temperature.py         # deterministic vs creative sampling
├── gemini_thinking.py            # Gemini 2.5 Flash thinking budget
├── fireworks_text.py             # basic OpenAI-compatible call
├── fireworks_conversation.py     # multi-turn conversation
├── fireworks_structured.py       # JSON output
├── fireworks_temperature.py      # temperature comparison
├── fireworks_thinking.py         # DeepSeek thinking-mode
├── openai_text.py                # OpenAI Responses API
├── openai_search.py              # web-search tool
├── tests/
│   ├── conftest.py               # sys.path shim
│   └── test_smoke.py             # AST-parse (no API keys needed)
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
| `make gemini-text` / `gemini-conversation` / `gemini-image` / `gemini-structured` / `gemini-temperature` / `gemini-thinking` | Gemini demos |
| `make fireworks-text` / `fireworks-conversation` / `fireworks-structured` / `fireworks-temperature` / `fireworks-thinking` | Fireworks demos |
| `make openai-text` / `openai-search` | OpenAI demos |

## Testing notes

- None of the scripts guard their work behind `if __name__ == "__main__":` — importing them would fire real API calls. `test_smoke.py` `ast.parse`s each script instead: catches syntax errors without needing keys or hitting the network.
- If you want a live end-to-end test, run the corresponding `make` target with the right env var set.
- If you refactor a script into a `main()` guarded by `if __name__ == "__main__":`, promote its entry in `test_smoke.py` from a parse test to a real import test.

## Typing discipline

- `pyrefly.toml` is on `preset = "strict"` with `python-version = "3.14"`.
- `google-genai` and `openai` SDK stubs use `Optional[...]` returns liberally. Expect narrowing / cast fixups at call sites; annotate incrementally as you touch each file.
