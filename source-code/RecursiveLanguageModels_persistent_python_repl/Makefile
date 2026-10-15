clean:
	rm -rf *~ *.sqlite __pycache__ .venv

create_db:
	uv run benchmark_suite.py --seed-only

run_omlx:
	uv run benchmark_suite.py --base-url http://127.0.0.1:8000/v1 --model Laguna-XS-2.1-6bit
