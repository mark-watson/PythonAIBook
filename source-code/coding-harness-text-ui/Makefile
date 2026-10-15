.PHONY: run sync check clean

run:
	uv run harness_tui.py

sync:
	uv sync

check:
	uv run python -m py_compile harness_tui.py agent_bridge.py

clean:
	rm -rf __pycache__ .venv
