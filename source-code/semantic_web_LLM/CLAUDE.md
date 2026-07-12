# semantic_web_LLM

Semantic Web QA: an LLM (Fireworks-hosted) extracts named entities from a natural-language question, SPARQL queries pull structured facts from DBpedia and/or Wikidata, and the LLM synthesizes the final answer. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

Set `FIREWORKS_API_KEY` for actual queries (see `README.md`).

## Layout

```
semantic_web_LLM/
├── library.py                 # shared LLM + SPARQL utilities
├── DBPedia.py                 # QA over DBpedia
├── Wikidata.py                # QA over Wikidata
├── DBPedia_and_Wikidata.py    # federated QA across both
├── tests/
│   ├── conftest.py
│   └── test_smoke.py          # imports only
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

- All four modules are safe to import: the three CLI scripts guard their entry points behind `if __name__ == "__main__":`, and `library.py` only defines functions at module level.
- No SPARQL or LLM calls fire during `just check`.

## Typing discipline

- `pyrefly.toml`: `preset = "strict"`, `python-version = "3.13"` (matched to `pyproject.toml`).
