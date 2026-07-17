.PHONY: check clean test similarity summarize zeroshot

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

similarity:
	uv run python sentence_similarity.py

summarize:
	uv run python summarization.py

zeroshot:
	uv run python zero_shot_classification.py

check:
	python3 -m py_compile *.py
	uv run python -c "import sentence_transformers, transformers"
