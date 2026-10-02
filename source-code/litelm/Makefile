.PHONY: check clean test run install uninstall build text streaming conversation tools embeddings vision thinking structured search image nvidia

# The library is a real package, so it can be installed into any uv
# environment. `uv sync` (and therefore `uv run`) installs it editable
# automatically; these targets are for pointing *another* project at it.
#
#   make install     # editable install into this project's .venv
#   make uninstall   # remove it again
#
# From a sibling project, the uv-native equivalent is:
#
#   uv add --editable ../litelm
install:
	uv pip install --editable .

uninstall:
	uv pip uninstall litelm

# Build the wheel/sdist that `make install` would install from PyPI.
build:
	uv build

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal
	rm -rf dist

test:
	uv run pytest -q

run:
	uv run python examples/example_text.py

text:
	uv run python examples/example_text.py

streaming:
	uv run python examples/example_streaming.py

conversation:
	uv run python examples/example_conversation.py

tools:
	uv run python examples/example_tools.py

embeddings:
	uv run python examples/example_embeddings.py

vision:
	uv run python examples/example_vision.py

thinking:
	uv run python examples/example_thinking.py

structured:
	uv run python examples/example_structured.py

search:
	uv run python examples/example_search.py

image:
	uv run python examples/example_image.py

nvidia:
	uv run python examples/example_nvidia.py

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pyrefly check
	uv run pytest -q
