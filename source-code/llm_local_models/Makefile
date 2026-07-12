.PHONY: clean test text streaming caching memory reasoning openai-compat image

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

text:
	uv run python ollama_text.py

streaming:
	uv run python ollama_streaming.py

caching:
	uv run python ollama_caching.py

memory:
	uv run python ollama_memory.py

reasoning:
	uv run python ollama_reasoning.py

openai-compat:
	uv run python ollama_openai_compat.py

image:
	uv run python image_to_text_description.py
