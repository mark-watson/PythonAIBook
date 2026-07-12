# TODO — symbolic-AI

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `lint` step (before ever reaching pyrefly), so no pyrefly numbers yet. `ruff format` reflowed 5 files. `pytest` passes (7 AST-parse tests) — see below.

## Ruff (7 errors, all in `hackernews.py`)

None are auto-fixable. All are legitimate code smells.

### 5× E722 — bare `except:`
`hackernews.py:29, 131, 139, 146, 151`

Replace each `except:` with `except Exception:` (or a more specific exception if known).

```python
try:
    spacy_model = spacy.load("en_core_web_sm")
except OSError:  # spaCy raises OSError when a model isn't installed
    from os import system
    system("python -m spacy download en_core_web_sm")
    spacy_model = spacy.load("en_core_web_sm")
```

The four in the Prolog-query block below are similar — narrow to whatever `swiplserver.PrologMQI` raises on query failure (probably `PrologError`).

### E711 — `!= None`
`hackernews.py:95`

```python
if story_json_data is not None and "url" in story_json_data:
```

### E402 — `import re` not at top of file
`hackernews.py:77`

Move `import re` up to the other imports at the top of the file.

## Pyrefly

Not yet run — blocked by the ruff step. Once the `hackernews.py` lint issues are fixed, run `just check` again to see the pyrefly baseline. Expect stub friction with `minizinc`, `swiplserver`, `spacy`, and `bs4`.

## Nice-to-have follow-ups

- `test_mzn.py` at the repo root is a MiniZinc demo script, not a pytest file. `pyproject.toml` sets `testpaths = ["tests"]` so it's not collected — but if you ever run `pytest test_mzn.py` explicitly it will try to run the demo. Consider renaming to `demo_mzn.py` at some point.
- `bw.py` requires `soar-sml`, which is not in `pyproject.toml`'s deps. It's mentioned in the file docstring — either add the dep or add a skip note in the README.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `7 passed` from pytest.
