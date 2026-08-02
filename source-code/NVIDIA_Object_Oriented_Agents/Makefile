.PHONY: run clean

# `uv run` reads the PEP 723 script header in travel_planner_agent.py
# and installs nooa + openai + pydantic into an ephemeral venv on first
# invocation. Subsequent runs reuse the cached environment.
run:
	uv run travel_planner_agent.py

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
