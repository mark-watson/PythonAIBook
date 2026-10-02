# fireworks_thinking.py - Extended reasoning with DeepSeek thinking mode
#
# DeepSeek models on Fireworks support a "thinking" mode that performs
# internal chain-of-thought reasoning before answering. The thinking
# tokens come back separately from the final answer, and litelm exposes
# them as response.reasoning.
#
# This example uses a classic logic puzzle to demonstrate thinking mode.
#
# Requirements: uv sync
# Environment: export FIREWORKS_API_KEY="your-api-key"
# Run: uv run python fireworks_thinking.py

import litelm

MODEL = "fireworks-ai/accounts/fireworks/models/deepseek-v4p1-flash"

prompt = """
A farmer has a fox, a chicken, and a bag of grain. He needs to cross
a river in a boat that can only carry him and one item at a time.
If left alone, the fox will eat the chicken, and the chicken will eat
the grain. How does the farmer get everything across safely?
"""

response = litelm.completion(
    MODEL,
    messages=[{"role": "user", "content": prompt}],
    extra={"thinking": {"type": "enabled"}},
)

# DeepSeek returns thinking tokens in the response when thinking is enabled
if response.reasoning:
    print("--- Thinking ---")
    print(response.reasoning)
    print("--- Answer ---")
print(response.content)
