# LLMs with Public APIs

The fastest way to use large language models is through cloud APIs. Google, OpenAI, and Anthropic all offer APIs that give you access to their most capable models with just a few lines of Python code. You don't need a GPU, you don't need to download model weights, and you can start building applications in minutes.

In this chapter we work through practical examples using the Google Gemini API and the OpenAI API. Both provide Python client libraries that handle authentication, request formatting, and response parsing. The patterns you learn here apply to other API providers as well — the core concepts of sending prompts, receiving completions, and managing conversations are the same across providers.

The examples for this chapter are in the directory **source-code/llm_public_apis**.

## Setup and Authentication

### Google Gemini

Google's Gemini models are accessed through the Google AI API using the **google-genai** Python SDK. You need a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

Install the SDK:

```bash
uv pip install google-genai
```

Store your API key in an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Here is the simplest possible example — send a prompt to Gemini and print the response:

```python
# gemini_text.py - Basic text generation with Google Gemini

import os
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Briefly explain what a transformer model is in AI."
)

print(response.text)
```

The output will be a concise explanation of transformer models. Each call to **generate_content** sends a request to Google's servers, which run the model and return the generated text.

### OpenAI

OpenAI's GPT models are accessed through the **openai** Python SDK. You need an API key from [OpenAI's platform](https://platform.openai.com/api-keys).

Install the SDK:

```bash
uv pip install openai
```

Store your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Here is the equivalent example using OpenAI:

```python
# openai_text.py - Basic text generation with OpenAI

from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

response = client.responses.create(
    model="gpt-5.4-nano",
    input="Briefly explain what a transformer model is in AI."
)

output_items = list(response.output)
for item in reversed(output_items):
    if getattr(type(item), "__name__", "") == "ResponseOutputMessage":
        for content in item.content:
            if type(content).__name__ == "ResponseOutputText":
                print(content.text)
                break
```

Both APIs follow the same pattern: create a client, send a prompt, and extract the generated text from the response.


## Text Generation

Text generation is the most fundamental LLM capability. You provide a prompt and the model generates a continuation or response.

### Controlling Output with Temperature

The **temperature** parameter controls how creative or deterministic the output is. A temperature of 0 produces the most predictable output (the model always picks the highest-probability next token). Higher temperatures (up to 1.0 or 2.0) produce more varied and creative output.

```python
# gemini_temperature.py - Effect of temperature on text generation

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
```

For most practical applications — code generation, data extraction, question answering — use a low temperature (0.0 to 0.3). For creative writing and brainstorming, higher temperatures (0.7 to 1.5) produce more interesting results.


## Thinking Models

Some models can engage in extended internal reasoning before producing a response. Google's Gemini 2.5 Flash supports a **thinking budget** that controls how much computation the model devotes to reasoning through the problem before answering.

```python
# gemini_thinking.py - Using Gemini's thinking mode for complex reasoning

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
    )
)

print(response.text)
```

The thinking budget is specified in tokens. A budget of 0 disables thinking entirely (useful for simple tasks where speed matters). Higher budgets allow the model to reason through more complex problems but increase latency and cost.


## Multi-Turn Conversations

Real applications often involve multi-turn conversations where the model needs to remember previous exchanges. Both APIs support this by passing conversation history with each request.

```python
# gemini_conversation.py - Multi-turn conversation with Gemini

import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Build a conversation as a list of content parts
conversation = []

def chat(user_message):
    """Send a message and get a response, maintaining conversation history."""
    conversation.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=conversation
    )
    conversation.append(
        types.Content(role="model", parts=[types.Part(text=response.text)])
    )
    return response.text

# A multi-turn conversation
print(chat("What is the capital of France?"))
print(chat("What is its population?"))  # "its" refers to Paris from context
print(chat("What are the top 3 tourist attractions there?"))
```

Notice that the second and third messages use pronouns ("its", "there") that only make sense given the conversation history. The model resolves these references correctly because it sees the full conversation with each request.


## Multimodal Input: Analyzing Images

Modern LLMs can process images alongside text. This enables applications like image description, document analysis, chart reading, and visual question answering.

```python
# gemini_image.py - Analyzing an image with Gemini

from google import genai
from google.genai import types
from PIL import Image
import os

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Load an image from disk
image = Image.open("photo.jpg")

prompt = "Describe what you see in this image. Be specific about people, objects, and setting."

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[prompt, image],
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0  # no thinking needed for simple description
        )
    )
)

print(response.text)
```

The key detail is that **contents** accepts a list containing both text and image objects. The model processes them together, understanding the image in the context of the text prompt.


## Web Search with LLMs

Some API providers allow the model to search the web as part of generating a response, which gives it access to current information beyond its training data.

Here is an example using OpenAI's web search tool:

```python
# openai_search.py - Web search with OpenAI

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4-nano",
    tools=[{"type": "web_search_preview"}],
    input="What were the major AI announcements at Google I/O 2025?"
)

output_items = list(response.output)
for item in reversed(output_items):
    kind = getattr(type(item), "__name__", "")
    if kind == "ResponseOutputMessage" and getattr(item, "role", None) == "assistant":
        for content in item.content:
            if type(content).__name__ == "ResponseOutputText":
                print(content.text)
                break
```

The **tools** parameter tells the model it can use web search to answer the question. The model decides whether to search based on the query — factual questions about recent events will trigger a search, while questions about well-known topics may not.

Google's Gemini also supports grounding with Google Search through a similar mechanism. Refer to the Google AI documentation for the current syntax, as this feature is actively evolving.


## Structured Output

For many applications you need the model to return data in a specific format — JSON, CSV, or a particular schema. LLMs can be instructed to produce structured output through careful prompting.

```python
# gemini_structured.py - Getting structured JSON output from Gemini

import os
import json
from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = """Extract the following information from the text below and return
it as a JSON object with keys: "name", "company", "role", "years_experience".

Text: "Jane Smith has been working as a Senior Data Scientist at Acme Corp
for the past 7 years. She specializes in NLP and recommendation systems."
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.0)
)

# Parse the JSON from the response
result = json.loads(response.text.strip().removeprefix("```json").removesuffix("```").strip())
print(json.dumps(result, indent=2))
```

Using temperature 0.0 is important for structured output — you want the model to be deterministic and precise rather than creative. Some APIs also support specifying a JSON schema directly in the request, which guarantees the output conforms to a specific structure.


## Practical Considerations

### Cost

API calls are billed per token. Input tokens (your prompt) and output tokens (the model's response) are priced separately, with output tokens typically costing 2-4x more. Prices vary significantly between providers and models:

- Smaller, faster models (Gemini 2.5 Flash, GPT-5.4-nano) are very inexpensive — often under $0.10 per million input tokens
- Frontier models (Gemini 2.5 Pro, GPT-5.4, Claude Opus) cost 10-50x more but offer superior reasoning

For most applications, start with a fast, inexpensive model and only upgrade to a frontier model for tasks that require it.

### Rate Limits

All API providers enforce rate limits — maximum requests per minute, tokens per minute, and tokens per day. Free tiers have lower limits. If you're building a production application, you'll need to implement retry logic with exponential backoff and consider batching requests where possible.

### Latency

API calls involve network round-trips and model inference time. Simple completions with small models return in under a second. Complex reasoning with frontier models can take 10-30 seconds or more. For interactive applications, consider streaming responses (both Gemini and OpenAI support this) so users see output as it's generated rather than waiting for the complete response.

### Privacy

Any data you send to an API is transmitted to the provider's servers. For sensitive data — medical records, financial information, proprietary code — review the provider's data usage policies carefully. Some providers offer data residency guarantees and opt-out options for training. For maximum privacy, consider using local models instead, as covered in the next chapter.

### Error Handling

API calls can fail for many reasons: network errors, rate limiting, content filtering, malformed requests, or service outages. Production code should handle these gracefully:

```python
import time

def generate_with_retry(client, prompt, max_retries=3):
    """Call the Gemini API with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
```


## Summary

Using LLMs through public APIs is the fastest path from idea to working application. The core pattern is simple across all providers: create a client, send a prompt, process the response. The richness comes from features like multi-turn conversations, multimodal input, web search, structured output, and thinking modes.

The main tradeoffs of the API approach are cost (per-token pricing), privacy (data leaves your machine), and dependence on the provider's availability. For applications where these tradeoffs are acceptable, public APIs give you access to the most capable models available.

In the next chapter we cover the alternative approach: running open-weights models locally on your own hardware, which offers privacy, no per-token cost, and offline operation at the expense of model capability and the need for suitable hardware.
