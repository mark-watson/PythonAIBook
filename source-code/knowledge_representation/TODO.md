# TODO — knowledge_representation

Outstanding warnings from the initial dev-setup pass. `just check` currently fails at the `typecheck` step. `ruff format/check` are clean and `pytest` passes (4 tests).

## Pyrefly (3 errors)

### 1. `dbpedia_cities.py:42` — `results.get("results", {}).get("bindings", [])` on unnarrowed SPARQL result
`SPARQLWrapper.query().convert()` returns a union of `Document | Graph | bytes | str | dict`, so pyrefly can't statically know `.get()` exists. We call it in JSON mode, so casting is safe:

```python
from typing import cast, Any

results = cast(dict[str, Any], sparql.query().convert())
bindings = results.get("results", {}).get("bindings", [])
```

### 2. `dbpedia_cities.py:52` — return type is `list[dict[str, str]]` but populations end up as `int`
```python
cities.append({
    "uri": b["city_uri"]["value"],
    "name": b["dbpedia_label"]["value"],
    "population": int(b["population"]["value"]),   # <- int
    "country": b["country_label"]["value"],
})
```

Fix: change the declared return type to accept mixed values:

```python
def fetch_cities() -> list[dict[str, str | int]]:
    ...
```

or (nicer) return a `TypedDict`:

```python
from typing import TypedDict

class CityRecord(TypedDict):
    uri: str
    name: str
    population: int
    country: str

def fetch_cities() -> list[CityRecord]: ...
```

### 3. `wikidata_person.py:37` — same `results.get()` issue as #1
Apply the same `cast(dict[str, Any], ...)` fix.

## Nice-to-have follow-ups

- `sqlite_lib.connection()` doesn't set `row_factory = sqlite3.Row`, but `sqlite_knowledge.py` sets it locally on its own connection. The test suite (`tests/test_smoke.py::test_sqlite_lib_roundtrip`) indexes rows positionally to work around this. If we ever add more sqlite_lib callers, consider making `sqlite_lib.connection()` set `row_factory = sqlite3.Row` by default.

## How to verify a fix

```bash
just check    # fmt-check + lint + typecheck + test
```

Should end with `INFO 0 errors` from pyrefly and `4 passed` from pytest.
