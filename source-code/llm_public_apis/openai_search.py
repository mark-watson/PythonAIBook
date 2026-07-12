# openai_search.py - Web search with OpenAI
#
# Demonstrates using OpenAI's web search tool to answer questions
# that require current information beyond the model's training data.
# The model decides whether to search based on the query content.
#
# Adapted from the gpt-5.4-nano-tests example.
#
# Requirements: uv pip install openai
# Environment: export OPENAI_API_KEY="your-api-key"

from openai import OpenAI
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

client = OpenAI()

# The web_search_preview tool lets the model search for current information
response = client.responses.create(
    model="gpt-5.4-nano",
    tools=[{"type": "web_search_preview"}],
    input="What were the major AI announcements at Google I/O 2025?",
)

# Extract the assistant's text from the response output
for item in response.output:
    if isinstance(item, ResponseOutputMessage) and item.role == "assistant":
        for content in item.content:
            if isinstance(content, ResponseOutputText):
                print(content.text)
                break
