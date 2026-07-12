"""Syntax smoke tests.

Both scripts run their MDP/Q-learning work at module top level (no
`if __name__ == "__main__":` guard). To avoid instantiating a
`gymnasium` env or running Q-learning during `just check`, we
`ast.parse` each script instead of importing it.
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = ["frozen_lake_qlearning.py", "mdp_demo.py"]

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_parses(script: str) -> None:
    source = (ROOT / script).read_text(encoding="utf-8")
    ast.parse(source, filename=script)
