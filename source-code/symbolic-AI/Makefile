.PHONY: check clean test family n-queens frame hackernews us-states test-mzn bw

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

family:          # needs swi-prolog + swiplserver
	uv run python family.py

n-queens:        # needs swi-prolog + swiplserver
	uv run python n_queens.py

hackernews:      # needs swi-prolog, spaCy en_core_web_sm, network
	uv run python hackernews.py

us-states:       # needs minizinc + coinbc solver
	uv run python us_states.py

test-mzn:        # needs minizinc + coinbc solver
	uv run python test_mzn.py

frame:           # pure Python
	uv run python frame.py

bw:              # needs soar-sml (not in project deps)
	uv run python bw.py

check:
	python3 -m py_compile *.py
	uv run python -c "import bs4, minizinc, soar_sml, spacy, swiplserver"
