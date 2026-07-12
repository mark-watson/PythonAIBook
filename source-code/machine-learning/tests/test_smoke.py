"""Syntax smoke tests.

Both scripts (`classification.py` and `load_data.py`) do their work at
module top level — they call `pd.read_csv(...)` and print results when
imported. So we `ast.parse` each script instead of importing it: this
catches syntax errors without needing the CSV data present.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = ["classification.py", "load_data.py"]

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_parses(script: str) -> None:
    source = (ROOT / script).read_text(encoding="utf-8")
    ast.parse(source, filename=script)
