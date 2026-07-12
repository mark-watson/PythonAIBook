# deep_learning_nlp

Three self-contained deep-learning NLP demos (sentence similarity, summarization, zero-shot classification) using the HuggingFace `transformers` and `sentence-transformers` libraries. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
deep_learning_nlp/
├── sentence_similarity.py       # sentence-transformers cosine similarity ranking
├── summarization.py             # facebook/bart-large-cnn summarizer
├── zero_shot_classification.py  # DeBERTa-v3 zero-shot classifier
├── tests/
│   ├── conftest.py              # sys.path shim
│   └── test_smoke.py            # import-only (no model downloads)
├── pyproject.toml
├── pyrefly.toml
├── justfile
├── Makefile
└── .claude/
    ├── settings.json
    └── hooks/{py-check.sh,py-stop.sh}
```

## Workflow rules

After any Python edit, `.claude/hooks/py-check.sh` runs automatically — it formats the file with ruff, applies safe autofixes, then typechecks it with pyrefly. Fix any reported errors before moving on.

When Claude finishes a turn, `.claude/hooks/py-stop.sh` runs the full gate (`ruff format --check`, `ruff check`, `pyrefly check`, `pytest`). If it fails, Claude must fix the errors before the session ends.

## Tools

| Command | What it does |
|---------|-------------|
| `just check` | Full gate: fmt-check + lint + typecheck + test |
| `just fmt` / `just lint` / `just typecheck` / `just test` | Individual steps |
| `make similarity` | Run the sentence-similarity demo |
| `make summarize` | Run the BART summarization demo |
| `make zeroshot` | Run the zero-shot classification demo |

## Typing discipline

- `pyrefly.toml` is on `preset = "strict"` with `python-version = "3.11"` (matched to `pyproject.toml`; torch 2.11 is the constraint).
- Third-party types come from `transformers` / `sentence-transformers` stubs. Any lints there tend to be optional-return or callable-narrowing issues — annotate at the call site as needed.

## Testing notes

- Tests use the `sys.path` shim in `tests/conftest.py` to import scripts by name from the repo root.
- Smoke tests are import-only. All three scripts guard their real work under `if __name__ == "__main__":`, so no HuggingFace models are downloaded at test time.
