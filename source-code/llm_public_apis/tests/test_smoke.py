"""Syntax smoke tests.

Every script in this project runs its work at *module level* — importing
them would call `client = genai.Client(...)` / `OpenAI()` and fire off API
requests, requiring live keys. Instead we `ast.parse` each script, which
verifies it's syntactically valid without executing anything.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = [
    "fireworks_conversation.py",
    "fireworks_structured.py",
    "fireworks_temperature.py",
    "fireworks_text.py",
    "fireworks_thinking.py",
    "gemini_conversation.py",
    "gemini_image.py",
    "gemini_structured.py",
    "gemini_temperature.py",
    "gemini_text.py",
    "gemini_thinking.py",
    "openai_search.py",
    "openai_text.py",
]

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_parses(script: str) -> None:
    source = (ROOT / script).read_text(encoding="utf-8")
    ast.parse(source, filename=script)
