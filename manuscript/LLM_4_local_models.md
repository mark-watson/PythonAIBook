# LLMs with Local Models

Running language models on your own hardware gives you privacy, zero per-token cost, and the ability to work offline. The tradeoff is that local models are generally smaller and less capable than the frontier models available through cloud APIs, and running larger models requires significant GPU memory or Apple Silicon unified memory.

In this chapter we focus on [Ollama](https://ollama.com), the most popular tool for running local models. Ollama handles model downloading, quantization, GPU acceleration, and exposes a simple API; you can go from zero to a running local LLM in minutes. We also briefly mention alternative tools at the end of the chapter.

If you want to go deeper into Ollama, including tool use, agents, RAG, and advanced configuration, see my book [Ollama in Action](https://leanpub.com/ollama-in-action).

The examples for this chapter are in the directory **source-code/llm_local_models**.

{width: "80%"}
![Architecture diagram for the LLM Local Models example](FIG_llm_local_models.jpg)


## Installing Ollama

Ollama is available for macOS, Linux, and Windows. On macOS:

```bash
brew install ollama
```

Or download the installer from [ollama.com](https://ollama.com). After installation, start the Ollama service:

```bash
ollama serve
```

This starts a local server on port 11434. The service runs in the background and manages model loading, GPU memory, and request handling.


## Downloading and Running Models

Ollama uses a Docker-like model for pulling and running models. To download a model:

```bash
ollama pull llama3.2:3b
```

This downloads Meta's Llama 3.2 with 3 billion parameters, quantized to about 2 GB. You can interact with it immediately from the command line:

```bash
ollama run llama3.2:3b "What is the capital of France?"
```

Some recommended models to start with:

| Model | Size | Strengths |
|-------|------|-----------|
| llama3.2:3b | 2 GB | Fast, good general purpose |
| gemma3:4b | 3 GB | Google's small model, strong reasoning |
| qwen3:4b | 2.6 GB | Excellent multilingual and coding |
| deepseek-r1:7b | 4.7 GB | Strong reasoning with explicit chain-of-thought |
| llava:7b | 4.7 GB | Vision model, can analyze images |


## Using Ollama from Python

As in the previous chapter, we drive the model through [**litellm**](https://github.com/BerriAI/litellm), the open-source library that puts a single OpenAI-format interface on more than 100 providers. The local server is simply another provider: `ollama_chat/llama3.2:3b` names the model, and switching to a cloud model means changing that string rather than the code.

```bash
uv sync   # installs litellm and the chapter's dependencies
```

### Basic Text Generation

The simplest use of litellm: send a prompt and print the response:

```python
import litellm

MODEL = "ollama_chat/llama3.2:3b"

response = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": "Briefly explain what a neural network is."}],
)
assert isinstance(response, litellm.ModelResponse), "Expected a non-streaming response"

print(response.choices[0].message.content)
```

This is the same shape as the cloud examples from the previous chapter — a provider-prefixed model string, a message list, and a chat completion response — but the request never leaves your machine.

### Streaming Responses

For interactive applications, streaming lets users see output as it's generated rather than waiting for the complete response:

```python
import litellm

MODEL = "ollama_chat/llama3.2:3b"

stream = litellm.completion(
    model=MODEL,
    messages=[{"role": "user", "content": "Write a short poem about programming."}],
    stream=True,
)
assert isinstance(stream, litellm.CustomStreamWrapper), "Expected a streaming response"

# Print each chunk as it arrives, without newlines between chunks
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()  # final newline
```

Each chunk carries an incremental piece of the response in **`chunk.choices[0].delta.content`**. The **flush=True** argument ensures text appears immediately rather than being buffered.


## Reasoning with Local Models

Some local models support explicit chain-of-thought reasoning, where the model shows its thinking process before providing a final answer. DeepSeek-R1 is particularly good at this.

First pull the model:

```bash
ollama pull deepseek-r1:7b
```

Here is an example that extracts both the reasoning trace and the final answer:

```python
import litellm


def reason_about(
    question: str, model: str = "ollama_chat/deepseek-r1:7b"
) -> dict[str, str]:
    """Ask a question and extract both reasoning and final answer."""
    response = litellm.completion(
        model=model, messages=[{"role": "user", "content": question}]
    )
    assert isinstance(response, litellm.ModelResponse), (
        "Expected a non-streaming response"
    )
    content = response.choices[0].message.content or ""

    # A separate reasoning field if the provider sent one, otherwise the
    # <think>...</think> block DeepSeek-R1 leaves inline.
    reasoning = getattr(response.choices[0].message, "reasoning_content", None) or ""
    answer = content
    if "<think>" in content and "</think>" in content:
        reasoning = content.split("<think>")[1].split("</think>")[0].strip()
        answer = content.split("</think>")[1].strip()

    return {"reasoning": reasoning, "answer": answer}


question = (
    "A bakery sells 3 types of bread. Each type comes in 2 sizes. "
    "How many different bread options are available? "
    "Respond with just the number and a brief explanation."
)

result = reason_about(question)

if result["reasoning"]:
    print("=== Reasoning ===")
    print(result["reasoning"])
    print()

print("=== Answer ===")
print(result["answer"])
```

The code checks two places for the reasoning trace. Models served by Ollama
usually leave it inline in the content as a `<think>...</think>` block, which we
split out by hand; providers that return the trace as a separate field populate
**`reasoning_content`** on the message, and litellm surfaces that for you via
`getattr(response.choices[0].message, "reasoning_content", None)`. Checking both
is the portable habit: the model's reasoning shows each step of its thinking,
making the output more transparent and debuggable than a black-box answer. This
is especially valuable for math, logic, and planning tasks.


## Conversation Memory with Ollama

Cloud APIs handle conversation history by passing the full message list with each request. With local models the same pattern applies, but since there are no per-token costs, you can maintain longer conversations without worrying about expense.

Here is an example that maintains a conversation with memory across multiple exchanges and uses a system prompt to shape the assistant's personality:

```python
from typing import Any

import litellm


class LocalAssistant:
    """A simple conversational assistant that maintains message history."""

    def __init__(
        self, model: str = "ollama_chat/llama3.2:3b", system_prompt: str = ""
    ) -> None:
        self.model = model
        self.messages: list[dict[str, Any]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining conversation history."""
        self.messages.append({"role": "user", "content": user_message})
        response = litellm.completion(model=self.model, messages=self.messages)
        assert isinstance(response, litellm.ModelResponse), (
            "Expected a non-streaming response"
        )
        reply = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def message_count(self) -> int:
        """Return the number of messages in the conversation history."""
        return len(self.messages)


# Create an assistant with a specific personality
assistant = LocalAssistant(
    system_prompt="You are a concise technical writing assistant. "
    "Keep answers under 3 sentences."
)

# Multi-turn conversation — the model remembers prior context
print("Q:", "What is gradient descent?")
print("A:", assistant.chat("What is gradient descent?"))
print()

print("Q:", "How does the learning rate affect it?")
print("A:", assistant.chat("How does the learning rate affect it?"))
print()

print("Q:", "What happens if I set it too high?")
print("A:", assistant.chat("What happens if I set it too high?"))
print()

print(f"(Conversation has {assistant.message_count()} messages)")
```

Note that unlike cloud APIs, keeping long conversation histories in local models is free; there are no per-token costs. The main constraint is the model's context window size, which varies by model (typically 4K to 128K tokens).


## Prompt Caching for Performance

When you send the same long context (a document, a knowledge base, or a detailed system prompt) with multiple questions, Ollama can cache the prompt processing to dramatically speed up subsequent requests. This happens automatically when the prefix of the prompt is identical across requests.

Here is an example that demonstrates the speedup:

```python
import secrets
import time

import litellm

MODEL = "ollama_chat/llama3.2:3b"

# keep_alive is an Ollama extension to the OpenAI-compatible body: it holds the
# model (and therefore the cached prompt prefix) in memory between requests.
OLLAMA_KEEP_ALIVE = "60m"

# A per-run nonce keeps the first request genuinely cold: this exact prefix has
# never been through the server before, so only the second request can hit the
# cache. Without it the benchmark would depend on what ran earlier in the day.
_RUN_NONCE = secrets.token_hex(8)

# A long static context that stays the same across queries
CONTEXT = f"[run {_RUN_NONCE}]\n" + (
    """
The Python programming language was created by Guido van Rossum and first
released in 1991. Python's design philosophy emphasizes code readability
with its notable use of significant whitespace. Python is dynamically typed
and garbage-collected. It supports multiple programming paradigms, including
structured, object-oriented, and functional programming.

Python consistently ranks as one of the most popular programming languages.
It is widely used in web development, data science, machine learning,
automation, and scientific computing. The language's large standard library
and extensive ecosystem of third-party packages make it suitable for a
wide range of applications.
"""
    * 20
)  # repeat to create a substantial context


def timed_query(question: str, label: str) -> float:
    """Send a query with the shared context; report the wall time it took."""
    start = time.time()
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": f"{CONTEXT}\n\nQuestion: {question}"}],
        keep_alive=OLLAMA_KEEP_ALIVE,
    )
    elapsed = time.time() - start
    assert isinstance(response, litellm.ModelResponse), (
        "Expected a non-streaming response"
    )

    # litellm's ModelResponse carries a usage object at runtime, but the class
    # itself does not declare the attribute, so read it defensively.
    usage = getattr(response, "usage", None)
    prompt = usage.prompt_tokens if usage else None
    if prompt is None:
        print(f"[{label}] Wall time: {elapsed:.2f}s | token counts not reported")
    else:
        print(f"[{label}] Wall time: {elapsed:.2f}s | {prompt} prompt tokens")
    return elapsed


# First request: cold start, processes the full context
time_a = timed_query("When was Python created?", "Cold start")

# Second request: same context prefix, different question — cache hit
time_b = timed_query("What paradigms does Python support?", "Cache hit")

if time_a > 0 and time_b > 0:
    print(f"\nWall-clock speedup on the warm request: {time_a / time_b:.1f}x")
```

A run looks like this: both requests carry the same long context, and the warm
one returns noticeably faster because it reuses the cached prefix:

```
[Cold start] Wall time: 1.44s | 2507 prompt tokens
[Cache hit] Wall time: 0.56s | 2510 prompt tokens

Wall-clock speedup on the warm request: 2.6x
```

The key settings for prompt caching:

- **keep_alive**: passed straight to `litellm.completion(...)` as a keyword argument, this Ollama extension keeps the model and its KV cache in memory between requests. A long duration like `"60m"` avoids paying the load cost again.
- **Identical prefix**: the cached portion must match exactly. If even one character of the context changes, the cache is invalidated — which is also why the example prepends a fresh nonce on each run, so the first request really is cold.
- **Measure the hit in wall-clock time**: Ollama's native API reports `prompt_eval_duration` and cached-prompt tokens, but litellm's `response.usage` exposes only `prompt_tokens`, so the cache hit has to be inferred from timing. The example wraps each call in `time.time()` and reports the seconds elapsed, and the warm request lands well under the cold one. Timing is noisy, so run it a few times before drawing conclusions.

Prompt caching is especially valuable for applications like document Q&A, where you load a long document once and then answer many questions about it.


## Image to Text Description (Vision Models)

Ollama also supports vision models, allowing you to pass an image along with your text prompt so the model can analyze the visual content. You just need to ensure you're using a vision-capable model (like `llava` or `qwen3.5`), inline the image as a base64 data URL, and send it as an OpenAI-style `image_url` content part.

Here is the sample image we will use for this example:

{width: "50%"}
![Sample ticket image used as input for the vision model](FIG_ticket.png)

Here is an example of asking a vision model to describe an image:

```python
import base64
import mimetypes
from pathlib import Path

import litellm

# Specify the path to the image file to be analyzed
image_path = Path("ticket.png")

# Inline the image as a data URL the vision models can decode
mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
image_data_url = (
    f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
)

# Send the image to the vision-capable model for a detailed description
response = litellm.completion(
    model="ollama_chat/qwen3.5:0.8b",  # Ensure you use a vision-capable model
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ],
    think=False,  # Suppresses the <think> reasoning block
)
assert isinstance(response, litellm.ModelResponse), "Expected a non-streaming response"

# Print the model's descriptive analysis of the image
print(response.choices[0].message.content)
```

Here is abbreviated output from running this example:

```bash
$ uv run image_to_text_description.py
This image is an event ticket from **Northern Arizona
University (NAU)** for a performance called **"Fanfares
and Fireworks."**

**Event Details**
- **Event Name**: *Fanfares and Fireworks*
- **Event Date**: Friday, September 26, 2025
- **Time**: 7:30 PM (AZ)
- **Venue**: Ardrey Memorial Auditorium
- **Performance**: Flagstaff Symphony Orchestra

**Ticket Information**
- **Ticket Type**: *Early Bird Tickets* / *New Subscriber C3*
- **Ticket Price**: $53.00 (Service Fee: $0.00)
- **Section**: Main Level, Row M, Seat 31

A QR code is displayed on the right side of the ticket
for scanning at the venue.
```

Even this small 0.8B-parameter vision model extracts detailed structured information from the ticket image: event details, pricing, seating, and layout elements. This simple capability makes it easy to add image understanding to your local applications without needing complex computer vision pipelines.


## One Client for Every Provider

Ollama exposes an OpenAI-compatible API endpoint — the same protocol litellm speaks to every provider. The local server is therefore not a special case; it is simply the `ollama_chat` provider, and the code below runs unchanged against a cloud API:

```python
import os

import litellm

# LLM_MODEL rather than LITELLM_MODEL: litellm already owns the LITELLM_*
# environment namespace for its own settings.
MODEL = os.environ.get("LLM_MODEL", "ollama_chat/llama3.2:3b")

response = litellm.completion(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the difference between a list and a tuple in Python?",
        },
    ],
    temperature=0.7,
)
assert isinstance(response, litellm.ModelResponse), "Expected a non-streaming response"

print(f"model: {MODEL}")
print(response.choices[0].message.content)
```

Set **LLM_MODEL** to a cloud model to send the same request elsewhere:

```bash
LLM_MODEL=fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash \
    uv run python ollama_openai_compat.py
```

That one-variable switch is the whole point of routing every example through
litellm: you can prototype against a local model and move to a hosted one — for
speed, capability, or a larger context window — without rewriting the code.


## Alternative Tools for Running Local Models

While Ollama is the system I usually use for running local models, several alternatives exist:

- **llama.cpp**: The C++ inference engine that Ollama is built on. Use it directly if you need maximum control over quantization, batching, or want to embed inference in a C/C++ application. Available at [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp).

- **LM Studio**: A desktop application with a graphical interface for downloading, managing, and chatting with local models. Good for non-programmers or for quickly trying different models. Available at [lmstudio.ai](https://lmstudio.ai).

- **vLLM**: A high-performance inference server optimized for throughput. Best suited for serving models to multiple users in production. Requires more GPU memory but can handle many concurrent requests efficiently. Available at [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm).

- **Hugging Face Transformers**: The **transformers** Python library can load and run models directly. This gives you the most flexibility for fine-tuning and custom inference pipelines, but requires more setup and GPU memory management. Best for researchers and advanced users.

For most developers getting started with local models, Ollama provides the best balance of simplicity and capability.


## Hardware Considerations

The amount of memory you need depends on the model size:

| Model Parameters | Quantized Size | Minimum RAM/VRAM |
|-----------------|----------------|-------------------|
| 1-3B | 1-2 GB | 8 GB RAM |
| 7-8B | 4-5 GB | 16 GB RAM |
| 14B | 8-9 GB | 16 GB RAM |
| 32-70B | 18-40 GB | 32-64 GB RAM |

On macOS with Apple Silicon (M1/M2/M3/M4), models run on the GPU using unified memory, which means your total system RAM is also your GPU memory. A MacBook with 16 GB of RAM can comfortably run 7-8B parameter models, and 32 GB or more enables larger models.

On Linux and Windows, a dedicated NVIDIA GPU with sufficient VRAM provides the best performance. Models can also run on CPU only, but inference is significantly slower (roughly 5-10x slower than GPU for most models).


## Summary

Running LLMs locally with Ollama gives you a private, cost-free, offline-capable alternative to cloud APIs. The setup is straightforward: install Ollama, pull a model, and start making API calls from Python. Features like streaming, conversation memory, prompt caching, and reasoning models make local models practical for many real applications.

The main tradeoff is capability: the largest models that run locally (7-14B parameters on typical hardware) are less capable than frontier cloud models with hundreds of billions of parameters. For many tasks (code assistance, text summarization, data extraction, conversational interfaces), local models perform well enough, and the privacy and cost benefits make them the better choice.

## Optional Practice Problems

Here are some exercises to help you apply the concepts from this chapter and extend the existing code examples.

### 1. Interactive Streaming Chat Loop (Easy)
- **Objective**: Extend [ollama_streaming.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/llm_local_models/ollama_streaming.py) to create a command-line chat interface. Instead of a single hardcoded query, prompt the user for input in a loop, stream the model's responses to the console in real-time, and exit when the user types `exit` or `quit`.
- **Key Concepts**: Input loops, real-time output streaming with `flush=True`, basic text generation.

### 2. Context Window and History Management (Medium)
- **Objective**: Modify the [LocalAssistant](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/llm_local_models/ollama_memory.py#L17) class in [ollama_memory.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/llm_local_models/ollama_memory.py) to handle context limits. Implement a maximum history size (e.g., `N`$ messages). When the conversation history exceeds this limit, the assistant should discard the oldest user/assistant exchanges. However, make sure that the original system prompt is always preserved at the beginning of the message history.
- **Key Concepts**: System prompt preservation, sliding window list management, message history truncation.

### 3. Extracting Structured JSON from Vision Models (Medium)
- **Objective**: Modify [image_to_text_description.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/llm_local_models/image_to_text_description.py) to ask the local vision model to extract structured data from `ticket.png` as a raw JSON block. Instruct the model to return keys like `event_name`, `date`, `time`, `venue`, and `price`. In your Python code, parse the model's output using Python's built-in `json` module, print the resulting dictionary, and handle any parsing errors.
- **Key Concepts**: Vision model prompting, structured output format directives, output parsing with JSON.

### 4. Multi-Model Answer Verification (Hard)
- **Objective**: Create a Python script that implements a multi-model validation pipeline. First, use [reason_about](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/llm_local_models/ollama_reasoning.py#L18) from [ollama_reasoning.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/llm_local_models/ollama_reasoning.py) with `ollama_chat/deepseek-r1:7b` to solve a logic puzzle (such as a word riddle or math problem). Then, extract the `<think>` reasoning trace and final answer. Finally, query a smaller general-purpose model like `ollama_chat/llama3.2:3b` with litellm, passing it both the original question and the reasoning trace, and ask it to verify whether the final answer is logically correct based on the reasoning trace.
- **Key Concepts**: Multi-model collaboration, chain-of-thought verification, automated self-correction/grading.

