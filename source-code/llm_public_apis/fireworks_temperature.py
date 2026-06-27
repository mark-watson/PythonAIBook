# fireworks_temperature.py - Effect of temperature on text generation
#
# Temperature controls randomness in the model's output:
#   0.0 = deterministic, always picks the most likely token
#   1.0+ = more creative, more varied output
#
# This script generates the same prompt at two different temperatures
# so you can see the difference in output style.
#
# Requirements: uv pip install openai
# Environment: export FIREWORKS_API_KEY="your-api-key"

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

prompt = "Write a one-sentence tagline for a coffee shop."

# Low temperature: deterministic, predictable
response_low = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)
print(f"Temperature 0.0: {response_low.choices[0].message.content}")

# High temperature: creative, varied
response_high = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5,
)
print(f"Temperature 1.5: {response_high.choices[0].message.content}")
