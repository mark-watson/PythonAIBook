# symbolic-AI

Classic symbolic-AI demos, each backed by a different external solver: SWI-Prolog for logic programming, MiniZinc for constraint satisfaction, Soar for cognitive architecture, plus a pure-Python frame system. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

External tool installs needed for full demos (see `README.md`):
- `brew install swi-prolog` (Prolog demos)
- `brew install minizinc` (constraint-satisfaction demos)
- `soar-sml` (Soar demo — not in `pyproject.toml`)

## Layout

```
symbolic-AI/
├── family.py         # Prolog family / grandparent inference (needs swiplserver)
├── n_queens.py       # 8-queens via CLP(FD) in Prolog
├── hackernews.py     # spaCy NER → Prolog facts pipeline
├── frame.py          # Lisp-style frame knowledge structures (pure Python)
├── us_states.py      # 4-color map colouring via MiniZinc
├── test_mzn.py       # tiny sum/product MiniZinc demo (NOT a pytest file)
├── bw.py             # Soar blocks-world (needs soar-sml, not in deps)
├── family.pl / n_queens.pl / us_states.mzn / test_mzn.mzn / bw.soar
├── tests/
│   ├── conftest.py
│   └── test_smoke.py # AST-parse each Python script
├── pyproject.toml    # testpaths = ["tests"] (keeps pytest away from test_mzn.py)
├── pyrefly.toml
├── justfile
├── Makefile
└── .claude/
    ├── settings.json
    └── hooks/{py-check.sh,py-stop.sh}
```

## Workflow rules

`.claude/hooks/py-check.sh` runs after every edit. `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. `just check` runs the same gate manually.

## Testing notes

- `test_mzn.py` at the repo root looks like a pytest file, but it's a MiniZinc demo. `pyproject.toml` sets `testpaths = ["tests"]` so pytest confines discovery to `tests/` and never tries to collect the demo.
- Every script here runs its work at module top level and depends on an external tool (Prolog, MiniZinc, Soar, spaCy, Hacker News). Importing them from a test is a bad idea — `test_smoke.py` uses `ast.parse` to catch syntax errors without invoking any solver.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.14"`.
- Expect stub friction with `minizinc`, `swiplserver`, `spacy`, and `bs4` since they don't all ship complete type stubs.
