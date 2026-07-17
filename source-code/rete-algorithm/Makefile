.PHONY: check clean test medical hospital ecom portfolio network smart-home

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

medical:
	uv run python example_medical.py

hospital:
	uv run python example_hospital_triage.py

ecom:
	uv run python example_ecom_pricing.py

portfolio:
	uv run python example_portfolio_rebalance.py

network:
	uv run python example_network_security.py

smart-home:
	uv run python example_smart_home.py

check:
	python3 -m py_compile *.py
	uv run python -c "import rete"
