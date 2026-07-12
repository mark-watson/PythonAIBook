# regression_and_clustering

Two scikit-learn intros: linear regression on the California housing dataset, and K-Means clustering. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
regression_and_clustering/
├── regression.py            # sklearn LinearRegression on California Housing
├── clustering.py            # sklearn KMeans on the same features
├── tests/
│   ├── conftest.py
│   └── test_smoke.py        # import-only
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

- Both scripts are `if __name__ == "__main__":`-guarded, so importing them is safe.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.11"` (matched to `pyproject.toml`).
