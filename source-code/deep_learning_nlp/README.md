# Deep Learning NLP – Source Code

This directory contains example code for the **Natural Language Processing Using Deep Learning** chapter.

## Files

- **summarization.py** — Text summarization using the facebook/bart-large-cnn model.
- **zero_shot_classification.py** — Zero-shot text classification using DeBERTa-v3.
- **sentence_similarity.py** — Sentence embedding and cosine similarity using sentence-transformers.

## Setup

Uses [`uv`](https://docs.astral.sh/uv/) for dependency management and [`just`](https://just.systems/) as the task runner.

```bash
# uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
# just — the Rust task runner (do NOT install the Python "just" package from PyPI)
brew install just

uv sync
```

Models are downloaded automatically to `~/.cache/huggingface` on first run.

## Running

```bash
uv run python summarization.py
uv run python zero_shot_classification.py
uv run python sentence_similarity.py
# or
make summarize
make zeroshot
make similarity
```

## Development workflow

```bash
just check       # fmt-check + lint + typecheck + test
just fmt         # format all Python files
just lint        # ruff --fix
just typecheck   # pyrefly (strict preset)
just test        # pytest with testmon (fast)
just test-all    # full parallel pytest run
```

Under Claude Code, `.claude/hooks/py-check.sh` runs after every edit (format + lint + per-file typecheck) and `.claude/hooks/py-stop.sh` runs the full gate before the turn ends. See `CLAUDE.md` for the full workflow contract.

## Architecture

![Deep learning NLP tasks architecture: summarization, zero-shot classification, and sentence similarity](FIG_deep_learning_nlp.jpg)
