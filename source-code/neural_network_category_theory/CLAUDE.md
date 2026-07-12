# neural_network_category_theory

A pure-stdlib Python implementation of a fixed-depth (3-layer) categorical NN, from the same paper as [`../deep_learning_category_theory/`](../deep_learning_category_theory/) but with a simpler architecture. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
neural_network_category_theory/
├── neural_network_category_theory.py   # everything — one guarded module
├── tests/
│   ├── conftest.py                     # sys.path shim
│   └── test_smoke.py                   # sigmoid / dot / deriv checks
├── pyproject.toml
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

- The module is guarded by `if __name__ == "__main__":`, so importing it is safe. `test_smoke.py` exercises the small pure math helpers (`sigmoid`, `sigmoid_deriv`, `dot`) — if any of these regresses, the whole categorical demo breaks downstream.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.14"`.
- The module is already heavily annotated, so pyrefly should be close to clean out of the box.
