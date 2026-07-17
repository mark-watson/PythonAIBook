.PHONY: check clean test run regression clustering

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

run:
	uv run python regression.py

regression:
	uv run python regression.py

clustering:
	uv run python clustering.py

check:
	python3 -m py_compile *.py
	uv run python -c "import numpy, pandas, sklearn"
