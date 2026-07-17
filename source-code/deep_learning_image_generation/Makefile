.PHONY: check clean test run local gemini

clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .hypothesis -exec rm -rf {} +
	rm -f .testmondata .testmondata-shm .testmondata-wal

test:
	uv run pytest -q

run:
	uv run python image_generation.py

local:
	uv run python image_generation.py

gemini:
	uv run python gemini_image_generation.py

check:
	python3 -m py_compile *.py
	uv run python -c "import PIL, diffusers, google, torch"
