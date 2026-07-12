"""Smoke tests for the knowledge-representation scripts.

`dbpedia_cities.py`, `wikidata_person.py`, and `sqlite_knowledge.py` guard
their live work behind `if __name__ == "__main__":`, so importing them here
does not hit the DBPedia or Wikidata SPARQL endpoints. `sqlite_lib.py` is a
library module with no side effects. We also exercise its small `connection`
+ `query` helpers against an in-memory SQLite database.
"""

import sqlite_lib


def test_dbpedia_cities_imports() -> None:
    import dbpedia_cities

    assert hasattr(dbpedia_cities, "QUERY_STRING")


def test_wikidata_person_imports() -> None:
    import wikidata_person

    assert hasattr(wikidata_person, "QUERY_TEMPLATE")


def test_sqlite_knowledge_imports() -> None:
    import sqlite_knowledge

    assert callable(sqlite_knowledge.build_knowledge_base)


def test_sqlite_lib_roundtrip() -> None:
    conn = sqlite_lib.connection(":memory:")
    sqlite_lib.query(conn, "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    sqlite_lib.query(conn, "INSERT INTO t (name) VALUES (?)", ("hello",))
    rows = sqlite_lib.query(conn, "SELECT name FROM t")
    # sqlite_lib.connection() does not set row_factory, so rows are plain tuples.
    assert [r[0] for r in rows] == ["hello"]
