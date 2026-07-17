.PHONY: check clean test run classification load

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

run:
	uv run python classification.py

classification:
	uv run python classification.py

load:
	uv run python load_data.py

check:
	python3 -m py_compile *.py
	uv run python -c "import pandas, sklearn"
