"""Syntax smoke tests.

Every script in this project runs at module top level and depends on
external tools (SWI-Prolog, MiniZinc, Soar SML, spaCy, Hacker News).
None of those are safe to invoke from a unit test, so we `ast.parse` each
script instead of importing it. This catches Python-level syntax errors
without needing any solver or network.

Note: `test_mzn.py` at the repo root is a MiniZinc demo script — pytest
does not pick it up because `testpaths = ["tests"]` restricts collection
to the `tests/` directory.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = [
    "bw.py",
    "family.py",
    "frame.py",
    "hackernews.py",
    "n_queens.py",
    "test_mzn.py",
    "us_states.py",
]

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_parses(script: str) -> None:
    source = (ROOT / script).read_text(encoding="utf-8")
    ast.parse(source, filename=script)
