# deep_learning_basics

A PyTorch feedforward network (9 → 15 → 15 → 1) that classifies the Wisconsin Breast Cancer dataset. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
deep_learning_basics/
├── cancer_model.py     # data loading + CancerNet + training loop
├── tests/
│   ├── conftest.py     # sys.path shim so tests can import cancer_model
│   └── test_model.py   # forward-pass + trainable-param smoke tests
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

Run `just check` manually at any time to verify the whole project.

## Tools

| Command | What it does |
|---------|-------------|
| `just fmt` | Format all Python files |
| `just lint` | Lint and autofix all Python files |
| `just typecheck` | Run pyrefly on the whole project |
| `just test` | Fast test run (testmon — only affected tests) |
| `just test-all` | Full parallel test run |
| `make run` | Run cancer_model.py (needs `../machine-learning/*.csv`) |

## Typing discipline

- `pyrefly.toml` is on `preset = "strict"` with `python-version = "3.11"` (matched to pyproject).
- The existing code has no annotations, so a fresh `just typecheck` will produce many errors. torch stubs are typically what surface first — start there.
- Config keys are **hyphenated** (`python-version`, not `python_version`).
- Unknown error-kind keys in `[errors]` silently break the config — add them one at a time.

## Testing notes

- Tests import `cancer_model` via the `sys.path` shim in `tests/conftest.py`.
- `test_model.py` only exercises the `CancerNet` module on synthetic input — it does **not** touch the CSVs under `../machine-learning/`, so the suite is self-contained.
- If you add tests that need real data, add a `@pytest.mark.skipif(...)` guard so the suite still passes when the CSVs are absent.
