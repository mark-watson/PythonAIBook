.PHONY: clean test gemini-text gemini-conversation gemini-image gemini-structured gemini-temperature gemini-thinking fireworks-text fireworks-conversation fireworks-structured fireworks-temperature fireworks-thinking openai-text openai-search

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

# ── Gemini
gemini-text:
	uv run python gemini_text.py

gemini-conversation:
	uv run python gemini_conversation.py

gemini-image:
	uv run python gemini_image.py

gemini-structured:
	uv run python gemini_structured.py

gemini-temperature:
	uv run python gemini_temperature.py

gemini-thinking:
	uv run python gemini_thinking.py

# ── Fireworks (via OpenAI-compatible endpoint)
fireworks-text:
	uv run python fireworks_text.py

fireworks-conversation:
	uv run python fireworks_conversation.py

fireworks-structured:
	uv run python fireworks_structured.py

fireworks-temperature:
	uv run python fireworks_temperature.py

fireworks-thinking:
	uv run python fireworks_thinking.py

# ── OpenAI
openai-text:
	uv run python openai_text.py

openai-search:
	uv run python openai_search.py
