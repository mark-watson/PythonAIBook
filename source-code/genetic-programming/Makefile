.PHONY: check clean test run regression logic regex plants

run: regression logic regex plants

regression:  # pure Python (no deps)
	uv run python example_1_symbolic_regression.py

logic:       # pure Python (no deps)
	uv run python example_2_boolean_logic.py

regex:       # pure Python (no deps)
	uv run python example_3_regex_synthesis.py

plants:      # pure Python (no deps)
	uv run python example_4_lsystem_plants.py

test:
	uv run pytest -q

check:
	uv run python -m py_compile *.py
	uv run ruff format --check .
	uv run ruff check .
	uv run pyrefly check
	uv run pytest -q

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal
