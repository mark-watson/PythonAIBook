# Source Code

Example code for the *Python AI Book*. Each subdirectory is a self-contained project — its own `pyproject.toml`, `uv.lock`, and per-chapter demos. See the `README.md` inside each project for what it does and how to run it.

## Dev environment

Every project is wired to the same [`uv`](https://docs.astral.sh/uv/)-based dev workflow. Two prerequisites once, machine-wide:

```bash
# uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# just — the Rust task runner (do NOT install the Python "just" package from PyPI)
brew install just
```

Then, inside any project:

```bash
uv sync
just check       # fmt-check + lint + typecheck + test — the full gate
just fmt         # ruff format .
just lint        # ruff check --fix .
just typecheck   # pyrefly check (strict preset)
just test        # pytest with pytest-testmon (fast — only affected tests)
just test-all    # full parallel pytest run (pytest-xdist)
make run         # or a project-specific target — see the project's Makefile
```

Each project also ships two Claude Code hooks:

- **`.claude/hooks/py-check.sh`** — runs after every `Write` / `Edit` / `MultiEdit` on a `.py` file: `ruff format` → `ruff check --fix` → `pyrefly check` on just that file. Exits non-zero on type errors so Claude sees the failure and can fix it in-place.
- **`.claude/hooks/py-stop.sh`** — runs when Claude finishes a turn: the full `just check` gate. Blocks the turn from ending if the project is red.

`CLAUDE.md` in each project documents that project's specific workflow contract, testing strategy, and any per-project quirks (e.g., `symbolic-AI/test_mzn.py` is a MiniZinc demo, not a pytest file).

## Per-project layout

Most projects are flat: source files at the repo root, tests under `tests/`, hooks under `.claude/hooks/`.

Two are structured differently and it's worth flagging:

- **`rete-algorithm/`** — proper package layout (`rete/`). Existing tests live at `rete/tests/test_engine.py`; `pyproject.toml` sets `testpaths = ["rete/tests"]`.
- **`Chess_game/`** — flat but with three cooperating modules (`chess_engine.py`, `chess_bot.py`, `main.py`) plus a `tests/` dir.

## Typing discipline

All projects run [`pyrefly`](https://github.com/facebook/pyrefly) on `preset = "strict"`. The Python version in `pyrefly.toml` matches the `requires-python` in each project's `pyproject.toml` (mostly 3.14, some on 3.13 / 3.12 / 3.11 depending on runtime deps).

Many of the projects predate this setup and were written without annotations. Where the initial `just check` doesn't pass green, that project has a **`TODO.md`** at the repo root listing the outstanding pyrefly errors with paste-ready fix snippets and a suggested annotation order. Working through those is incremental — no rush.

## Testing strategy

Which testing approach a project uses depends on how its scripts are shaped:

- **Import smoke tests** — where every script guards its work behind `if __name__ == "__main__":`, tests just `import` each module and assert the entry point is callable. Cheap, fast, and catches import-time bugs.
- **AST-parse tests** — where scripts run their work at module top level (LLM API calls, live Ollama chats, SPARQL queries, MDP training), importing them would fire real requests or start heavy jobs. Instead we `ast.parse` each script — proves it's syntactically valid without executing anything.
- **Real unit tests** — where a project ships genuine library code (`Chess_game/chess_engine.py`, `deep_learning_category_theory.py`, `neural_network_category_theory.py`, `knowledge_representation/sqlite_lib.py`, `rete-algorithm/rete/`), tests exercise the pure helpers directly.

## Updating an individual project

Any file change inside a project is gated by that project's hooks — you'll get instant feedback from ruff and pyrefly, and won't be able to end a Claude turn while `just check` is red. To run the full gate manually:

```bash
cd source-code/<project>
just check
```

To bypass the end-of-turn hook once (rarely needed):

```bash
# just don't — fix the errors instead. That's the whole point of the gate.
```

## Reference

The dev-env pattern used here comes from `../Claude_Code_language_integrations/python_test/`, which is the minimal reference implementation.
