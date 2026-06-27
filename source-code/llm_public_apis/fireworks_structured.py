# fireworks_structured.py - Getting structured JSON output from Fireworks
#
# Demonstrates extracting structured data from unstructured text.
# The model is prompted to return a JSON object, and we parse it
# in Python. Using temperature=0 ensures deterministic output.
#
# Requirements: uv pip install openai
# Environment: export FIREWORKS_API_KEY="your-api-key"

import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

prompt = """Extract the following information from the text below and return
it as a JSON object with keys: "name", "company", "role", "years_experience".

Text: "Jane Smith has been working as a Senior Data Scientist at Acme Corp
for the past 7 years. She specializes in NLP and recommendation systems."
"""

response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)

# Parse the JSON from the response (strip any markdown code fences)
raw = response.choices[0].message.content.strip()
raw = raw.removeprefix("```json").removesuffix("```").strip()
result = json.loads(raw)
print(json.dumps(result, indent=2))
