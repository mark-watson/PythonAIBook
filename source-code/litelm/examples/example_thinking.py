"""Reasoning traces from thinking models.

Two shapes show up in practice, and litelm exposes both:

* a *separate* reasoning field -- Fireworks' DeepSeek thinking mode returns it
  next to the answer, and litelm surfaces it as ``response.reasoning``;
* an *inline* ``<think>...</think>`` block in the content -- what DeepSeek-R1
  served by Ollama does, exactly as ``llm_local_models/ollama_reasoning.py``
  extracts it.

Run::

    ollama pull deepseek-r1:7b
    LITELM_MODEL=ollama/deepseek-r1:7b uv run example_thinking.py
    LITELM_MODEL=fireworks-ai/accounts/fireworks/models/deepseek-v4-flash \\
        uv run example_thinking.py
"""

import os
import re

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/deepseek-r1:7b")
PROMPT = """
A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in
a boat that can only carry him and one item at a time. If left alone, the fox
will eat the chicken, and the chicken will eat the grain. How does the farmer
get everything across safely? Answer briefly.
"""

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extras(model: str) -> dict[str, object] | None:
    """Fireworks/DeepSeek thinking mode is requested with an extra body field."""
    if model.startswith("fireworks-ai/"):
        return {"thinking": {"type": "enabled"}}
    return None


def split_inline_thinking(content: str) -> tuple[str | None, str]:
    """Split a ``<think>...</think>`` preamble from the answer."""
    match = _THINK.search(content)
    if match is None:
        return None, content
    return match.group(1).strip(), content[match.end() :].strip()


def main() -> None:
    response = litelm.completion(MODEL, PROMPT, extra=extras(MODEL))

    reasoning = response.reasoning
    answer = response.content or ""
    inline, answer = split_inline_thinking(answer)
    reasoning = reasoning or inline

    if reasoning:
        print("=== Reasoning ===")
        print(reasoning)
        print()
    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
