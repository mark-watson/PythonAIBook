# Regression and Clustering – Source Code

This directory contains example code for the **Regression and Clustering** chapter.

## Running

```bash
uv run regression.py
uv run clustering.py
```

## Files

- **regression.py** — Linear and polynomial regression on the California Housing dataset
- **clustering.py** — K-Means clustering on the Iris dataset with silhouette score evaluation

## Architecture

![Regression and clustering pipeline architecture](FIG_regression_and_clustering.jpg)

## Development workflow

Uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://just.systems/) as the task runner. Install both, then:

```bash
uv sync
just check       # fmt-check + lint + typecheck + test
just fmt         # ruff format
just lint        # ruff --fix
just typecheck   # pyrefly (strict)
just test        # pytest with testmon (fast)
just test-all    # full parallel pytest run
```

Under Claude Code, `.claude/hooks/py-check.sh` runs after every edit (format + lint + per-file typecheck) and `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. See `CLAUDE.md` for the workflow contract.
