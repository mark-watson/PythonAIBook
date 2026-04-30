# Knowledge Representation – Source Code

This directory contains example code for the **Knowledge Representation with Graph and Relational Databases** chapter.

## SPARQL Examples (Wikidata and DBPedia)

Query public knowledge bases using SPARQL:

```bash
uv run wikidata_person.py
uv run dbpedia_cities.py
```

## SQLite Examples

Knowledge representation using relational databases (no external dependencies):

```bash
uv run sqlite_knowledge.py
```

## Files

- **wikidata_person.py** — Query Wikidata for information about a person (birth place, occupations)
- **dbpedia_cities.py** — Query DBPedia for cities with population data
- **sqlite_lib.py** — Reusable SQLite helper functions
- **sqlite_knowledge.py** — Knowledge base with entity/relationship tables and JOIN queries
