"""Web search through the Responses API.

``llm_public_apis/openai_search.py`` hand-builds an OpenAI client around
``client.responses.create(tools=[{"type": "web_search_preview"}])``. litelm's
:func:`~litelm.responses` speaks that protocol directly, and the tool itself is
the :data:`litelm.WEB_SEARCH` constant -- the model searches on the provider's
side and litelm hands back the answer that cites the results.

Run::

    export OPENAI_API_KEY="sk-..."
    uv run example_search.py
"""

import os

import litelm

MODEL = os.environ.get("LITELM_MODEL", "openai/gpt-5.4-nano")
QUESTION = "What were the major AI announcements at Google I/O 2025?"


def main() -> None:
    response = litelm.responses(MODEL, QUESTION, tools=[litelm.WEB_SEARCH])
    print(f"model: {MODEL}")
    print(f"status: {response.finish_reason}\n")
    print(response.content)


if __name__ == "__main__":
    main()
