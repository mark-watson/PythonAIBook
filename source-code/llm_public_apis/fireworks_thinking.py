# fireworks_thinking.py - Extended reasoning with DeepSeek thinking mode
#
# DeepSeek models on Fireworks support a "thinking" mode that performs
# internal chain-of-thought reasoning before answering. The thinking
# tokens come back separately from the final answer, and litellm exposes
# them as ``message.reasoning_content``.
#
# This example uses a classic logic puzzle to demonstrate thinking mode.
#
# Requirements: uv sync
# Environment: export FIREWORKS_API_KEY="your-api-key"
# Run: uv run python fireworks_thinking.py

import litellm

MODEL = "fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash"

prompt = """
A farmer has a fox, a chicken, and a bag of grain. He needs to cross
a river in a boat that can only carry him and one item at a time.
If left alone, the fox will eat the chicken, and the chicken will eat
the grain. How does the farmer get everything across safely?
"""

response = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    thinking={"type": "enabled"},
)

# DeepSeek returns thinking tokens in the response when thinking is enabled
assert isinstance(response, litellm.ModelResponse)
message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None)
if reasoning:
    print("--- Thinking ---")
    print(reasoning)
    print("--- Answer ---")
print(message.content)
