"""Tool calling: Python functions the model can ask litelm to run.

Tools are ordinary functions taking one ``dict`` of arguments, so the same
weather stub that a local model calls also works against OpenAI, Gemini,
Fireworks and NVIDIA.

``completion`` *returns* tool calls instead of running them. The manual loop
below makes that explicit; :func:`litelm.chat_with_tools` is the one-line
version of the same loop.

Run::

    uv run example_tools.py
    LITELM_MODEL=openai/gpt-5.4-nano uv run example_tools.py
"""

import ast
import operator
import os
from typing import Any

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/llama3.2:3b")
QUESTION = "What is the weather in Paris, and what is 12 * 7 + 5?"

_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def get_weather(args: dict[str, Any]) -> str:
    """A stub: a real tool would call a weather service."""
    return f"sunny and 22C in {args['location']}"


def calculator(args: dict[str, Any]) -> str:
    """Evaluate a small arithmetic expression, safely."""
    return str(_evaluate(ast.parse(str(args["expression"]), mode="eval").body))


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return float(_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


TOOLS = [
    litelm.make_tool(
        "get_weather",
        "Get the current weather for a location",
        [("location", "string", "City name, e.g. Paris")],
        get_weather,
    ),
    litelm.make_tool(
        "calculator",
        "Evaluate an arithmetic expression",
        [("expression", "string", "Expression to evaluate, e.g. 12 * 7 + 5")],
        calculator,
    ),
]


def main() -> None:
    # The loop written out, so the request / execute / reply protocol is visible.
    messages: list[litelm.Message] = [{"role": "user", "content": QUESTION}]
    for step in range(1, 6):
        response = litelm.completion(MODEL, messages, tools=TOOLS)
        if not response.tool_calls:
            print(f"answer: {response.content}")
            return
        print(f"step {step}: model asked for {len(response.tool_calls)} tool call(s)")
        for call in response.tool_calls:
            print(f"  -> {call.name}({dict(call.arguments)})")
        messages.append(litelm.assistant_message(response))
        for result in litelm.execute_tool_calls(TOOLS, response.tool_calls):
            print(f"  <- {result.result}")
            messages.append(litelm.tool_message(result))
    print("gave up after 5 steps")


if __name__ == "__main__":
    main()
