.PHONY: check clean test mdp qlearning

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

mdp:
	uv run python mdp_demo.py

qlearning:
	uv run python frozen_lake_qlearning.py

check:
	python3 -m py_compile *.py
	uv run python -c "import gymnasium, mdptoolbox, numpy"
