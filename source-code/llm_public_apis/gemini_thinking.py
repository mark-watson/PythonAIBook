# gemini_thinking.py - Using Gemini's thinking mode for complex reasoning
#
# Gemini 2.5 Flash supports a "thinking budget" that controls how much
# internal reasoning the model performs before answering. Higher budgets
# allow deeper reasoning but increase latency and cost.
#
# This example uses a classic logic puzzle to demonstrate thinking mode.
#
# Requirements: uv pip install google-genai
# Environment: export GOOGLE_API_KEY="your-api-key"

from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = """
A farmer has a fox, a chicken, and a bag of grain. He needs to cross
a river in a boat that can only carry him and one item at a time.
If left alone, the fox will eat the chicken, and the chicken will eat
the grain. How does the farmer get everything across safely?
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=1000  # allow up to 1000 tokens of reasoning
        )
    ),
)

print(response.text)
