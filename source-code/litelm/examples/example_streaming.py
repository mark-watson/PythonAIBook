"""Streaming responses through the uniform interface.

``stream=True`` turns the return value into an iterator of
:class:`litelm.StreamChunk`; ``chunk.text`` is the incremental delta. Each chunk
also carries ``finish_reason`` and a running snapshot of any ``tool_calls`` the
model is emitting.

Run::

    uv run example_streaming.py
    LITELM_MODEL=openai/gpt-5.4-nano uv run example_streaming.py
"""

import os

import litelm

MODEL = os.environ.get("LITELM_MODEL", "ollama/llama3.2:3b")
PROMPT = "Write a short poem about programming."


def main() -> None:
    chunks = 0
    for chunk in litelm.completion(MODEL, PROMPT, stream=True):
        print(chunk.text, end="", flush=True)
        chunks += 1
        if chunk.finish_reason is not None:
            print(f"\n\n(finish reason: {chunk.finish_reason})")
    print(f"({chunks} chunks)")


if __name__ == "__main__":
    main()
