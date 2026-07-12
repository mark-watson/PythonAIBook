# openknowledge_format

A single-script OKF explorer that reads a bundle of knowledge fragments and answers questions about it via a local Ollama model. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

Ollama must be running for the actual demo (see `README.md`).

## Layout

```
openknowledge_format/
├── okf_explorer.py       # main script (guarded by __main__)
├── bundle/               # example OKF bundle (data)
├── tests/
│   ├── conftest.py       # sys.path shim
│   └── test_smoke.py     # import-only
├── pyproject.toml
├── pyrefly.toml          # excludes bundle/**
├── justfile
├── Makefile
└── .claude/
    ├── settings.json
    └── hooks/{py-check.sh,py-stop.sh}
```

## Workflow rules

`.claude/hooks/py-check.sh` runs after every edit. `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. `just check` runs the same gate manually.

## Testing notes

- The script is `if __name__ == "__main__":`-guarded, so importing it is safe. `test_smoke.py` does a one-line import check.
- Pyrefly is configured to exclude the `bundle/` directory (no Python code inside).

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.12"` (matched to `pyproject.toml`).
