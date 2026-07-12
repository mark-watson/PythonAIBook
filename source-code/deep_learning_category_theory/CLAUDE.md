# deep_learning_category_theory

A pure-stdlib Python implementation of the five categorical-ML perspectives from Jia et al. (2025). One long, heavily-annotated file (`deep_learning_category_theory.py`) wired to a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
deep_learning_category_theory/
├── deep_learning_category_theory.py   # all five perspectives, ~1150 lines
├── tests/
│   ├── conftest.py                    # sys.path shim
│   └── test_math_helpers.py           # sigmoid / relu / dot / matvec / outer / transpose
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
| `make run` | Run the full demo (`uv run python deep_learning_category_theory.py`) |

## Typing discipline

- `pyrefly.toml` is on `preset = "strict"` with `python-version = "3.14"`.
- The source already has function signatures typed throughout, so pyrefly should be near-clean out of the box. If new lints surface, `[[wiki-link:TODO.md]]` catches them at the end of the initial pass.
- Config keys are **hyphenated** (`python-version`, not `python_version`).
- Unknown error-kind keys in `[errors]` silently break the config — add them one at a time.

## Testing notes

- Tests import the module via the `tests/conftest.py` `sys.path` shim.
- `test_math_helpers.py` covers the ten small pure helpers (`sigmoid`, `relu`, `dot`, `matvec`, `outer`, `transpose`, etc.) — a regression in any of these breaks every downstream demo, so this is the cheapest tripwire.
- Higher-level pieces (Para lenses, dropout, Bayesian layers, k-means, sheaf gluing) are exercised only when you run the full `make run` demo; add targeted tests as you edit them.
