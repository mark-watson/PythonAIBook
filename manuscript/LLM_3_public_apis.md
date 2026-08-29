# LLMs with Public APIs

The fastest way to use large language models is through cloud APIs. Google, OpenAI, and Anthropic,  all offer APIs that give you access to their most capable proprietary models with just a few lines of Python code. Fireworks.ai and NVIDIA are inference providers in the USA that offer fast inferencing for many open weight models. You don't need a GPU, you don't need to download model weights, and you can start building applications in minutes.

In this chapter we work through practical examples using the Google Gemini API, the OpenAI API, the Fireworks.ai API, and NVIDIA's free NIM inference service. Each provides Python client libraries (or compatibility layers) that handle authentication, request formatting, and response parsing. The patterns you learn here apply to other API providers as well; the core concepts of sending prompts, receiving completions, and managing conversations are the same across providers.

The examples for this chapter are in the directory **source-code/llm_public_apis**.

{width: "80%"}
![Architecture diagram for the LLM Public APIs example](FIG_llm_public_apis.jpg)

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

Here is the simplest possible example: send a prompt to Gemini and print the response:

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

### Fireworks.ai

Fireworks.ai provides fast, cost-effective access to open-weight models through an OpenAI-compatible API. This means you can use the **openai** Python SDK you already installed; just point it at Fireworks' endpoint. DeepSeek V4 Flash, the default model we use here, delivers strong performance at a fraction of the cost of proprietary models.

Get a free API key from [fireworks.ai](https://fireworks.ai/api-keys) and set it as an environment variable:

```bash
export FIREWORKS_API_KEY="your-api-key-here"
```

The simplest Fireworks example looks nearly identical to OpenAI's chat completions pattern:

```python
# fireworks_text.py - Basic text generation with Fireworks.ai

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

response = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[
        {"role": "user", "content": "Briefly explain what a transformer model is in AI."}
    ],
)

print(response.choices[0].message.content)
```

The output is a concise explanation of transformer models. The key difference from the standard OpenAI setup is the **base_url** parameter, which redirects the SDK to Fireworks' servers. The **messages** format uses the familiar Chat Completions API structure with role-based message objects.

### NVIDIA NIM (Free Inference)

NVIDIA's [build.nvidia.com](https://build.nvidia.com) service provides free API access to a broad catalogue of open-weight models (Llama, Mistral, Phi, DeepSeek, and NVIDIA's own Nemotron family) hosted on NVIDIA GPUs. Like Fireworks.ai, NVIDIA exposes an OpenAI-compatible endpoint, so the same **openai** SDK works with just a different **base_url**.

Sign up for a free account, generate a key, and store it in an environment variable:

```bash
export NVIDIA_API_KEY="your-api-key-here"
```

The **NVIDIA_client.py** example in this chapter's source directory takes a slightly different shape from the other examples. Instead of running its work at module level, it wraps the API in a small reusable library so you can **import** it from other scripts:

```python
# NVIDIA_client.py - Library for NVIDIA's free inference service

import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
_BASE_URL = "https://integrate.api.nvidia.com/v1"


def get_client() -> OpenAI:
    return OpenAI(
        base_url=_BASE_URL,
        api_key=os.getenv("NVIDIA_API_KEY"),
    )


def complete(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Single-turn prompt → reply."""
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty response from model")
    return content


def chat(
    messages: list[ChatCompletionMessageParam],
    model: str = DEFAULT_MODEL,
) -> str:
    """Multi-turn conversation history → next assistant reply."""
    response = get_client().chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Empty response from model")
    return content


if __name__ == "__main__":
    print(complete("Briefly explain what a transformer model is in AI."))
```

Running the module directly (**python NVIDIA_client.py**) executes the demo in the **__main__** block; importing it from another script gives you clean helper functions with no module-level side effects:

```python
from NVIDIA_client import complete, chat

# Single-turn: fire and forget
print(complete("Summarize the plot of Moby Dick in two sentences."))

# Multi-turn: pass along the conversation history yourself
history = []
for turn in ["What is the capital of France?", "What is its population?"]:
    history.append({"role": "user", "content": turn})
    reply = chat(history)
    history.append({"role": "assistant", "content": reply})
    print(reply)
```

Two things are worth noting about this structure. First, **get_client()** builds a fresh **OpenAI** instance on each call rather than a module-level singleton. That means importing the module never touches the network or requires **NVIDIA_API_KEY** to be set; the key is only read the first time you actually call one of the helper functions. This makes the module safe to import from tests and other utilities. Second, the **if __name__ == "__main__":** guard means the demo only runs when you execute the file directly; **import**ing it stays silent. That pattern, thin wrapper functions around a lazily constructed client guarded by a **__main__** block, is a good template to follow when moving from prototyping to any code you'll import elsewhere.

NVIDIA's model catalogue includes **meta/llama-3.1-8b-instruct** (fast and general-purpose, used as the default above), **mistralai/mixtral-8x7b-instruct-v0.1**, **nvidia/llama-3.1-nemotron-70b-instruct** (strong on reasoning and instruction-following), and many more. Change the **model** argument or the **DEFAULT_MODEL** constant to try a different one. Because the endpoint is OpenAI-compatible, everything else you learn in this chapter (temperature, structured output, multi-turn conversations) transfers directly.


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

For most practical applications (code generation, data extraction, question answering), use a low temperature (0.0 to 0.3). For creative writing and brainstorming, higher temperatures (0.7 to 1.5) produce more interesting results.

The Fireworks API works the same way. Since Fireworks uses the OpenAI Chat Completions format, temperature is passed as a top-level parameter:

```python
# fireworks_temperature.py - Effect of temperature on text generation

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

prompt = "Write a one-sentence tagline for a coffee shop."

response_low = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)
print(f"Temperature 0.0: {response_low.choices[0].message.content}")

response_high = client.chat.completions.create(
    model="accounts/fireworks/models/deepseek-v4-flash",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.5,
)
print(f"Temperature 1.5: {response_high.choices[0].message.content}")
```

You'll see the same pattern as Gemini: temperature 0.0 produces a safe, predictable tagline, while 1.5 yields something more surprising and original.


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

Fireworks' DeepSeek models also support thinking mode. When enabled, the model performs internal chain-of-thought reasoning before producing its final answer, and the reasoning tokens are returned separately:

```python
# fireworks_thinking.py - Extended reasoning with DeepSeek thinking mode

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

message = response.choices[0].message
if hasattr(message, "thinking") and message.thinking:
    print("--- Thinking ---")
    print(message.thinking)
    print("--- Answer ---")
print(message.content)
```

The **extra_body** parameter passes the thinking configuration directly to the Fireworks API. DeepSeek's approach differs from Gemini's thinking budget; instead of controlling how many tokens to spend on reasoning, you simply enable or disable thinking mode. The reasoning trace is available via `message.thinking`, which is useful for debugging and understanding the model's logic.


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

The same pattern works with Fireworks using the Chat Completions message list format. Instead of building Gemini Content/Part objects, you append plain dicts to the messages array:

```python
# fireworks_conversation.py - Multi-turn conversation with Fireworks

import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.getenv("FIREWORKS_API_KEY"),
)

messages = []

def chat(user_message):
    """Send a message and get a response, maintaining conversation history."""
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="accounts/fireworks/models/deepseek-v4-flash",
        messages=messages,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

print(chat("What is the capital of France?"))
print(chat("What is its population?"))
print(chat("What are the top 3 tourist attractions there?"))
```

The Fireworks implementation is notably simpler than the Gemini version; the Chat Completions format uses standard dicts for messages rather than typed objects, which makes message history management straightforward.


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

The **tools** parameter tells the model it can use web search to answer the question. The model decides whether to search based on the query; factual questions about recent events will trigger a search, while questions about well-known topics may not.

Google's Gemini also supports grounding with Google Search through a similar mechanism. Refer to the Google AI documentation for the current syntax, as this feature is actively evolving.


## Structured Output

For many applications you need the model to return data in a specific format: JSON, CSV, or a particular schema. LLMs can be instructed to produce structured output through careful prompting.

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

Using temperature 0.0 is important for structured output: you want the model to be deterministic and precise rather than creative. Some APIs also support specifying a JSON schema directly in the request, which guarantees the output conforms to a specific structure.

The Fireworks version uses the same prompting strategy. The code differs only in how the client is configured and how the response text is accessed:

```python
# fireworks_structured.py - Getting structured JSON output from Fireworks

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

raw = response.choices[0].message.content.strip()
raw = raw.removeprefix("```json").removesuffix("```").strip()
result = json.loads(raw)
print(json.dumps(result, indent=2))
```

The output is identical to the Gemini version: a clean JSON object with the extracted fields. The JSON cleanup logic (stripping markdown code fences) is the same because all LLMs tend to wrap code blocks in backticks.


## Practical Considerations

### Cost

API calls are billed per token. Input tokens (your prompt) and output tokens (the model's response) are priced separately, with output tokens typically costing 2-4x more. Prices vary significantly between providers and models:

- Smaller, faster models (Gemini 2.5 Flash, GPT-5.4-nano) are very inexpensive, often under $0.10 per million input tokens
- Frontier models (Gemini 2.5 Pro, GPT-5.4, Claude Opus) cost 10-50x more but offer superior reasoning

For most applications, start with a fast, inexpensive model and only upgrade to a frontier model for tasks that require it.

### Rate Limits

All API providers enforce rate limits: maximum requests per minute, tokens per minute, and tokens per day. Free tiers have lower limits. If you're building a production application, you'll need to implement retry logic with exponential backoff and consider batching requests where possible.

### Latency

API calls involve network round-trips and model inference time. Simple completions with small models return in under a second. Complex reasoning with frontier models can take 10-30 seconds or more. For interactive applications, consider streaming responses (both Gemini and OpenAI support this) so users see output as it's generated rather than waiting for the complete response.

### Privacy

Any data you send to an API is transmitted to the provider's servers. For sensitive data (medical records, financial information, proprietary code), review the provider's data usage policies carefully. Some providers offer data residency guarantees and opt-out options for training. For maximum privacy, consider using local models instead, as covered in the next chapter.

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

## Optional Practice Problems

To help solidify the concepts covered in this chapter, try implementing the following exercises. You can create these scripts in your local workspace to extend the existing code examples.

### 1. Easy: Dynamic Tagline Generator
Modify the `gemini_temperature.py` example to create a command-line script that:
* Prompts the user to enter a business type (e.g., "coffee shop", "dog walking service", "indie game studio").
* Prompts the user to enter a target audience (e.g., "college students", "busy professionals", "hardcore gamers").
* Generates three different taglines using three distinct temperature values (e.g., `0.0` for deterministic/professional, `0.7` for balanced, and `1.5` for highly creative/unconventional).
* Displays the temperature alongside the generated tagline so you can compare the direct effects of the temperature parameter on creativity.

### 2. Medium: CLI Chatbot with System Instructions
Using the `gemini_conversation.py` script as a starting point, build a fully interactive command-line chatbot:
* When the script starts, prompt the user to input a "persona" or system instructions (e.g., "You are a helpful assistant who answers exclusively in pirate speak" or "You are an encouraging coding mentor").
* Configure the client or prompt structure to enforce this persona. (Hint: In the Gemini API, you can pass system instructions via `types.GenerateContentConfig(system_instruction="...")`).
* Enter a loop that repeatedly prompts the user for input (`input("You: ")`).
* Exit the loop gracefully if the user types `exit` or `quit`.
* Print the assistant's responses and append each turn to the conversation history to maintain context.

### 3. Medium: Structured Multimodal Data Extractor
Combine the concepts from `gemini_image.py` and `gemini_structured.py` to extract structured information from a document image:
* Find or capture an image containing unstructured text (e.g., a photo of a restaurant receipt, a business card, or a book cover).
* Load the image using `Pillow` and write a script that sends the image along with a prompt requesting the model to extract key details.
* Instruct the model to return a structured JSON response (e.g., for a book cover, extract `"title"`, `"author"`, `"publisher"`, and `"estimated_publication_year"`).
* Parse the JSON response in Python and display the extracted keys and values in a formatted terminal printout.

### 4. Hard: High-Availability Structured Parser with Retry and Fallback
Create a robust text processing pipeline that extracts structural sentiment analysis from product reviews:
* Write a script that takes a list of raw user reviews (e.g., `"The battery life is amazing, but the screen is a bit dim."`).
* Define a target schema for the output containing: `sentiment` (must be one of `Positive`, `Negative`, or `Neutral`), `sentiment_score` (a float between `0.0` and `1.0`), and a list of `pros` and `cons`.
* Implement a function to call the Gemini API using `gemini-3-flash-preview` to perform this extraction, ensuring temperature is set to `0.0`.
* Integrate the exponential backoff retry logic described in the **Error Handling** section of this chapter. If a call fails, retry up to 3 times with progressive delays.
* **Add a Fallback Provider**: If the Gemini API call still fails after 3 retries (due to rate limits, quota limits, or API outage), catch the exception, print a warning, and fall back to the OpenAI API using `gpt-5.4-nano` to process that specific review.

