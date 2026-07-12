.PHONY: clean test dbpedia wikidata both

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

dbpedia:
	uv run python DBPedia.py "tell me about the following: IBM, Microsoft"

wikidata:
	uv run python Wikidata.py "tell me about the following: IBM, Microsoft"

both:
	uv run python DBPedia_and_Wikidata.py "tell me about the following: IBM, Microsoft"
