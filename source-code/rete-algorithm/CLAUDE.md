# rete-algorithm

A lightweight, idiomatic Python implementation of the Rete algorithm as a proper `rete/` package, with six worked-example scripts. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
rete-algorithm/
├── rete/                       # the package
│   ├── __init__.py             # public re-exports (Fact, Pat, Var, Cond, ReteEngine, …)
│   ├── alpha.py                # alpha network
│   ├── beta.py                 # beta network + join nodes
│   ├── conflict.py             # conflict set resolution
│   ├── context.py              # rule execution context
│   ├── engine.py               # ReteEngine driver
│   ├── facts.py                # Fact base + WM management
│   ├── network.py              # network builder
│   ├── patterns.py             # Pat / Var / Cond primitives
│   ├── tokens.py               # tokens flowing through the beta net
│   └── tests/
│       ├── __init__.py
│       └── test_engine.py      # existing engine test suite
├── example_*.py                # six domain examples (medical, hospital, e-com, portfolio, network security, smart home)
├── Design.md                   # design notes / architecture write-up
├── pyproject.toml              # testpaths = ["rete/tests"]
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

- Unlike the flat-layout projects here, `rete-algorithm` has a real Python package (`rete/`) with the existing test suite at `rete/tests/test_engine.py`. `pyproject.toml` sets `testpaths = ["rete/tests"]` to point pytest at that location.
- The six `example_*.py` scripts at repo root are runnable demos, not tests. They are included in pyrefly's checked set but not in `testpaths`.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.14"`.
- Both the package sources under `rete/` and the top-level `example_*.py` scripts are checked.
