"""Syntax smoke tests.

Every script in this project runs its work at *module level* (no
`if __name__ == "__main__":` guard), so importing them would spawn network
requests to a running `ollama` server. Instead we `ast.parse` each script,
which verifies it's syntactically valid without executing anything.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = [
    "image_to_text_description.py",
    "ollama_caching.py",
    "ollama_memory.py",
    "ollama_openai_compat.py",
    "ollama_reasoning.py",
    "ollama_streaming.py",
    "ollama_text.py",
]

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_parses(script: str) -> None:
    source = (ROOT / script).read_text(encoding="utf-8")
    ast.parse(source, filename=script)
