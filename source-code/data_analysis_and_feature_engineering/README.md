# Exploratory Data Analysis and Feature Engineering – Source Code

This directory contains example code for the **EDA and Feature Engineering** chapter.

## Files

- **eda.py** — Exploratory data analysis: summary statistics, correlations, missing values, and outlier detection on the California Housing dataset.
- **feature_engineering.py** — Creating derived features, one-hot encoding, missing data imputation, feature scaling, and measuring the impact on model performance.

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://just.systems/) as the task runner.

```bash
# uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# just — the Rust task runner (do NOT install the Python "just" package from PyPI)
brew install just
```

Then install the deps:

```bash
uv sync
```

## Running

```bash
uv run python eda.py
uv run python feature_engineering.py
# or
make eda
make features
```

## Development workflow

```bash
just check       # fmt-check + lint + typecheck + test
just fmt         # format all Python files
just lint        # ruff --fix
just typecheck   # pyrefly (strict preset)
just test        # pytest with testmon (fast, only affected tests)
just test-all    # full parallel pytest run
```

When used with Claude Code, `.claude/hooks/py-check.sh` runs after every edit (format + lint + per-file typecheck) and `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. See `CLAUDE.md` for the full workflow contract.

## Architecture

![Data analysis and feature engineering pipeline architecture](FIG_data_analysis_and_feature_engineering.jpg)
