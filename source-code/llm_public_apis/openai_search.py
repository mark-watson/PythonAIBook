# openai_search.py - Web search with OpenAI
#
# Demonstrates using OpenAI's web search tool to answer questions
# that require current information beyond the model's training data.
# The model decides whether to search based on the query content.
#
# The tool runs on OpenAI's side, so it is passed as a provider-side tool
# (litelm.WEB_SEARCH) rather than a Python function. litelm only sees the
# final answer, which carries the search results as annotations.
#
# Adapted from the gpt-5.4-nano-tests example.
#
# Requirements: uv sync
# Environment: export OPENAI_API_KEY="your-api-key"
# Run: uv run python openai_search.py

import litelm

MODEL = "openai/gpt-5.4-nano"

# The web_search_preview tool lets the model search for current information
response = litelm.responses(
    MODEL,
    "What were the major AI announcements at Google I/O 2025?",
    tools=[litelm.WEB_SEARCH],
)

print(response.content)
