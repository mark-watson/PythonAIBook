"""The example scripts must import cleanly and expose a ``main`` entry point.

Unlike the demos in ``llm_local_models`` and ``llm_public_apis`` -- which run
their work at module level and can therefore only be ``ast.parse``-d -- these
examples guard everything behind ``main()``, so importing them is safe and
proves the public interface is complete.

``examples/`` is a package, so the import path here is the same one a reader
would use: ``examples.example_text``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import litelm

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"

EXAMPLES = sorted(path.name for path in EXAMPLES_DIR.glob("example_*.py"))


def test_every_expected_example_is_present() -> None:
    assert EXAMPLES == [
        "example_conversation.py",
        "example_embeddings.py",
        "example_image.py",
        "example_nvidia.py",
        "example_search.py",
        "example_streaming.py",
        "example_structured.py",
        "example_text.py",
        "example_thinking.py",
        "example_tools.py",
        "example_vision.py",
    ]


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_parses(example: str) -> None:
    ast.parse((EXAMPLES_DIR / example).read_text(encoding="utf-8"), filename=example)


@pytest.mark.parametrize("example", EXAMPLES)
def test_example_imports_and_exposes_main(example: str) -> None:
    module = importlib.import_module(f"examples.{example.removesuffix('.py')}")
    assert callable(module.main)


def test_tool_definitions_in_the_tools_example_are_valid() -> None:
    example = importlib.import_module("examples.example_tools")
    names = {tool.name for tool in example.TOOLS}
    assert names == {"get_weather", "calculator"}
    schemas = litelm.tool_schemas(example.TOOLS)
    assert len(schemas) == 2


def test_the_tools_example_calculator_handles_arithmetic() -> None:
    example = importlib.import_module("examples.example_tools")
    assert example.calculator({"expression": "12 * 7 + 5"}) == "89.0"
    assert example.calculator({"expression": "-(2 + 3)"}) == "-5.0"


def test_the_tools_example_calculator_rejects_unknown_syntax() -> None:
    example = importlib.import_module("examples.example_tools")
    with pytest.raises(ValueError, match="unsupported expression"):
        example.calculator({"expression": "__import__('os')"})


def test_thinking_example_splits_inline_traces() -> None:
    example = importlib.import_module("examples.example_thinking")
    reasoning, answer = example.split_inline_thinking("<think>two plus two</think>4")
    assert reasoning == "two plus two"
    assert answer == "4"


def test_thinking_example_leaves_plain_content_alone() -> None:
    example = importlib.import_module("examples.example_thinking")
    assert example.split_inline_thinking("4") == (None, "4")


def test_structured_example_strips_a_json_fence() -> None:
    example = importlib.import_module("examples.example_structured")
    assert example.parse_json_object('```json\n{"name": "Jane"}\n```') == {
        "name": "Jane"
    }


def test_structured_example_rejects_non_objects() -> None:
    example = importlib.import_module("examples.example_structured")
    with pytest.raises(TypeError, match="expected a JSON object"):
        example.parse_json_object("[1, 2]")


def test_vision_example_only_sends_think_to_ollama() -> None:
    example = importlib.import_module("examples.example_vision")
    assert example.provider_extras("ollama/qwen3.5:0.8b") == {"think": False}
    assert example.provider_extras("gemini/gemini-3-flash-preview") is None


def test_embeddings_example_cosine_similarity() -> None:
    example = importlib.import_module("examples.example_embeddings")
    assert example.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert example.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert example.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_the_public_interface_is_exported() -> None:
    for name in litelm.__all__:
        assert hasattr(litelm, name), name
    assert len(set(litelm.__all__)) == len(litelm.__all__), "duplicate export"
