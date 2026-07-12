.PHONY: clean test dbpedia wikidata sqlite

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

dbpedia:
	uv run python dbpedia_cities.py

wikidata:
	uv run python wikidata_person.py

sqlite:
	uv run python sqlite_knowledge.py
