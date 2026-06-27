# fireworks_thinking.py - Extended reasoning with DeepSeek thinking mode
#
# DeepSeek models on Fireworks support a "thinking" mode that performs
# internal chain-of-thought reasoning before answering. The thinking
# tokens are returned separately from the final answer.
#
# This example uses a classic logic puzzle to demonstrate thinking mode.
#
# Requirements: uv pip install openai
# Environment: export FIREWORKS_API_KEY="your-api-key"

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

prompt = """
A farmer has a fox, a chicken, and a bag of grain. He needs to cross
a river in a boat that can only carry him and one item at a time.
If left alone, the fox will eat the chicken, and the chicken will eat
the grain. How does the farmer get everything across safely?
"""

response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    extra_body={"thinking": {"type": "enabled"}},
)

# DeepSeek returns thinking tokens in the response when thinking is enabled
message = response.choices[0].message
if hasattr(message, "thinking") and message.thinking:
    print("--- Thinking ---")
    print(message.thinking)
    print("--- Answer ---")
print(message.content)
