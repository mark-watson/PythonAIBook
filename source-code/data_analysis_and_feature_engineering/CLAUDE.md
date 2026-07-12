# data_analysis_and_feature_engineering

Two standalone scripts from the EDA / Feature Engineering chapter, wired to a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

Flat layout — both scripts live at the repo root:

```
data_analysis_and_feature_engineering/
├── eda.py                    # EDA on California Housing
├── feature_engineering.py    # Derived features + model comparison
├── tests/
│   ├── conftest.py           # puts repo root on sys.path
│   └── test_smoke.py         # import-only smoke tests
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
| `make eda` | Run the EDA script |
| `make features` | Run the feature-engineering script |

## Typing discipline

- `pyrefly.toml` is set to `preset = "strict"` with `python-version = "3.14"`.
- The two scripts have no type annotations yet, so a fresh `just typecheck` will produce errors. Third-party types come from numpy / pandas / sklearn stubs; expect some churn as those stubs evolve. Add annotations incrementally.
- Config keys are **hyphenated** (`python-version`, not `python_version`).
- Unknown error-kind keys in `[errors]` silently break the config — add them one at a time.

## Testing notes

- Tests import the scripts by name from repo root via the `tests/conftest.py` `sys.path` shim.
- `test_smoke.py` only *imports* the modules — the actual work is behind `if __name__ == "__main__":`, so no network fetch of the California Housing dataset happens during test runs.
- If you add real tests, prefer synthetic DataFrames over `fetch_california_housing()` to keep the suite offline and fast.
