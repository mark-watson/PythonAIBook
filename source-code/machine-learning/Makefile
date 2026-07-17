.PHONY: check clean test classification load

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

classification:
	uv run python classification.py

load:
	uv run python load_data.py

check:
	python3 -m py_compile *.py
	uv run python -c "import pandas, sklearn"
