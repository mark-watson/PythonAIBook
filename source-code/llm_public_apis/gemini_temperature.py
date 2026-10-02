# gemini_temperature.py - Effect of temperature on text generation
#
# Temperature controls randomness in the model's output:
#   0.0 = deterministic, always picks the most likely token
#   1.0+ = more creative, more varied output
#
# This script generates the same prompt at two different temperatures
# so you can see the difference in output style.
#
# Requirements: uv sync
# Environment: export GOOGLE_API_KEY="your-api-key"
# Run: uv run python gemini_temperature.py

import litellm

MODEL = "gemini/gemini-3-flash-preview"

prompt = "Write a one-sentence tagline for a coffee shop."

# Low temperature: deterministic, predictable
response_low = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)
assert isinstance(response_low, litellm.ModelResponse)
print(f"Temperature 0.0: {response_low.choices[0].message.content}")

# High temperature: creative, varied
response_high = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5,
)
assert isinstance(response_high, litellm.ModelResponse)
print(f"Temperature 1.5: {response_high.choices[0].message.content}")
