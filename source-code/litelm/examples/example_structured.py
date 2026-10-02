"""Structured JSON output, provider independently.

The trick from ``llm_public_apis/fireworks_structured.py`` and
``gemini_structured.py`` is unchanged -- ask for JSON, then strip any markdown
fence before parsing -- but the same code runs against a local model or a
cloud one.

Run::

    uv run example_structured.py
    LITELM_MODEL=fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash \\
        uv run example_structured.py
"""

import json
import os
from typing import Any

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/llama3.2:3b")

TEXT = (
    "Jane Smith has been working as a Senior Data Scientist at Acme Corp for "
    "the past 7 years. She specializes in NLP and recommendation systems."
)

PROMPT = f"""Extract the following information from the text below and return it
as a JSON object with keys: "name", "company", "role", "years_experience".

Text: "{TEXT}"
"""


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a ```json ... ``` fence."""
    raw = text.strip().removeprefix("```json").removesuffix("```").strip()
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise TypeError(f"expected a JSON object, got: {raw!r}")
    return decoded


def main() -> None:
    # temperature=0 keeps the extraction deterministic.
    response = litelm.completion(MODEL, PROMPT, temperature=0.0)
    print(json.dumps(parse_json_object(response.content or ""), indent=2))


if __name__ == "__main__":
    main()
