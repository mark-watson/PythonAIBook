# reinforcement_learning

Two RL demos: MDP value iteration via `pymdptoolbox`, and Q-learning on FrozenLake via `gymnasium`. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
reinforcement_learning/
├── mdp_demo.py              # pymdptoolbox — small hand-crafted MDP
├── frozen_lake_qlearning.py # gymnasium FrozenLake-v1 with Q-learning
├── tests/
│   ├── conftest.py
│   └── test_smoke.py        # AST-parse (no env init)
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

- Both scripts run their MDP setup / Q-learning loop at module top level (no `if __name__ == "__main__":` guard). Importing them would instantiate a `gymnasium` env and start training. `test_smoke.py` therefore uses `ast.parse` — proves each script parses cleanly without executing.
- If you wrap either script's top-level code in a `main()` guarded by `__name__`, promote its entry to a real import test.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.11"` (matched to `pyproject.toml`; `pymdptoolbox` doesn't publish stubs for 3.14).
