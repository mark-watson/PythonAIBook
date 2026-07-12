# machine-learning

Two-script sklearn intro: build a cancer-classification model and load the raw data. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
machine-learning/
├── classification.py       # scikit-learn Logistic Regression
├── load_data.py            # pandas → numpy CSV pipeline
├── labeled_cancer_data.csv # train split
├── labeled_test_data.csv   # test split
├── tests/
│   ├── conftest.py
│   └── test_smoke.py       # AST-parse (no CSV read needed)
├── pyproject.toml
├── pyrefly.toml
├── justfile
├── Makefile
└── .claude/
    ├── settings.json
    └── hooks/{py-check.sh,py-stop.sh}
```

## Workflow rules

`.claude/hooks/py-check.sh` runs after every edit (ruff format → autofix → per-file pyrefly). `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. `just check` runs the same gate manually.

## Testing notes

- Both scripts do their work at module top level (no `if __name__ == "__main__":` guard), so a real import would trigger the CSV load and model fit. `test_smoke.py` therefore uses `ast.parse` — proves each script parses cleanly without executing anything.
- If you refactor either into `main()` with a `__main__` guard, promote the corresponding entry to a real import test.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.11"` (matched to `pyproject.toml`).
