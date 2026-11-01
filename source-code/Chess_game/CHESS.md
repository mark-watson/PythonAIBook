# Chess_game

A pure-Python chess engine with a Negamax + alpha-beta search bot and an interactive CLI. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

Flat layout — the three source modules live at the repo root, tests live under `tests/`:

```
Chess_game/
├── chess_engine.py   # board, move generation, Zobrist hashing
├── chess_bot.py      # Negamax + TT + quiescence search + evaluation
├── main.py           # CLI game loop
├── tests/
│   ├── conftest.py   # puts repo root on sys.path so tests can import chess_engine
│   └── test_engine.py
├── pyproject.toml
├── pyrefly.toml
├── justfile
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
| `make run` | Launch the interactive CLI (`uv run python main.py`) |

## Typing discipline

- `pyrefly.toml` is set to `preset = "strict"` with `python-version = "3.14"`.
- The existing chess code was written before this dev setup and has no type annotations, so a fresh `just typecheck` will produce many errors. Add annotations incrementally — start at the leaves (`chess_engine.py` constants and small helpers) and work outward.
- Config keys are **hyphenated** (`python-version`, not `python_version`).
- Unknown error-kind keys in `[errors]` silently break the config — add them one at a time.

## Testing notes

- Tests live under `tests/` and import from the repo root via `tests/conftest.py`, which prepends the repo root to `sys.path`. No `src/` layout, no installable package.
- `test_engine.py` runs Perft to depth 3 (8,902 nodes at the starting position) and verifies the incremental Zobrist hash matches a fresh recomputation on every make/unmake — this is the tripwire for move-generation regressions.
