# knowledge_representation

Four small examples of representing structured knowledge: two SPARQL endpoints (DBPedia, Wikidata) and a relational KB in SQLite. Uses a uv-based dev workflow with strict pyrefly typing, ruff formatting/linting, pytest, and two Claude Code hooks that gate every edit and every turn end.

## Quick start

```bash
uv sync
just check   # fmt-check + lint + typecheck + test
```

## Layout

```
knowledge_representation/
├── dbpedia_cities.py       # DBPedia SPARQL query for city data
├── wikidata_person.py      # Wikidata SPARQL query about a person
├── sqlite_knowledge.py     # relational KB (in-memory sqlite3)
├── sqlite_lib.py           # tiny reusable connect/query helpers
├── tests/
│   ├── conftest.py         # sys.path shim
│   └── test_smoke.py       # imports + sqlite_lib roundtrip
├── pyproject.toml
├── pyrefly.toml
├── justfile
├── Makefile
└── .claude/
    ├── settings.json       # hook wiring
    ├── settings.local.json # pre-existing per-user permission overrides — preserved
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
| `make dbpedia` | Query DBPedia for city data (needs internet) |
| `make wikidata` | Query Wikidata for a person (needs internet) |
| `make sqlite` | Build and query the in-memory scientists KB |

## Typing discipline

- `pyrefly.toml` is on `preset = "strict"` with `python-version = "3.14"`.
- `SPARQLWrapper` has thin stubs — expect that some return-type narrowing may need explicit annotations at call sites.

## Testing notes

- `test_smoke.py` imports the three main scripts (each has an `if __name__ == "__main__":` guard, so no SPARQL requests fire during testing) and does a full sqlite_lib roundtrip (create table, insert, select) against an `:memory:` connection.
- The Wikidata / DBPedia scripts are network-bound; do not add tests that call them at check time.
