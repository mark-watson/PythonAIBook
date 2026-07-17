.PHONY: check clean test run

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

run:
	uv run python -m anomaly_detection.wisconsin

check:
	find . -name "*.py" -not -path "./.venv/*" -not -path "./tests/*" -exec python3 -m py_compile {} +
	uv run python -c "import anomaly_detection, beartype, matplotlib, numpy, sklearn"
