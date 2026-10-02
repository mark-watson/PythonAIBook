# gemini_thinking.py - Using Gemini's thinking mode for complex reasoning
#
# Gemini models reason internally before answering, and the OpenAI-compatible
# endpoint exposes the same knob OpenAI uses: reasoning_effort. litellm forwards
# it as an ordinary request parameter, and Gemini maps it onto its thinking
# configuration -- "low" is a ~1024-token budget on Gemini 2.5 and the
# "low" thinking level on Gemini 3. Reasoning cannot be switched off for
# Gemini 3 models, only reduced.
#
# This example uses a classic logic puzzle to demonstrate thinking mode.
#
# Requirements: uv sync
# Environment: export GOOGLE_API_KEY="your-api-key"
# Run: uv run python gemini_thinking.py

import litellm

MODEL = "gemini/gemini-3-flash-preview"

prompt = """
A farmer has a fox, a chicken, and a bag of grain. He needs to cross
a river in a boat that can only carry him and one item at a time.
If left alone, the fox will eat the chicken, and the chicken will eat
the grain. How does the farmer get everything across safely?
"""

response = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    reasoning_effort="low",  # keep the thinking budget small
)

assert isinstance(response, litellm.ModelResponse)
print(response.choices[0].message.content)
