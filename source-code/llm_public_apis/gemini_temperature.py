# gemini_temperature.py - Effect of temperature on text generation
#
# Temperature controls randomness in the model's output:
#   0.0 = deterministic, always picks the most likely token
#   1.0+ = more creative, more varied output
#
# This script generates the same prompt at two different temperatures
# so you can see the difference in output style.
#
# Requirements: uv pip install google-genai
# Environment: export GOOGLE_API_KEY="your-api-key"

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = "Write a one-sentence tagline for a coffee shop."

# Low temperature: deterministic, predictable
response_low = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.0)
)
print(f"Temperature 0.0: {response_low.text}")

# High temperature: creative, varied
response_high = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=1.5)
)
print(f"Temperature 1.5: {response_high.text}")
