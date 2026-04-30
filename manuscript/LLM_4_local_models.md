# LLMs with Local Models

Running language models on your own hardware gives you privacy, zero per-token cost, and the ability to work offline. The tradeoff is that local models are generally smaller and less capable than the frontier models available through cloud APIs, and running larger models requires significant GPU memory or Apple Silicon unified memory.

In this chapter we focus on [Ollama](https://ollama.com), the most popular tool for running local models. Ollama handles model downloading, quantization, GPU acceleration, and exposes a simple API — you can go from zero to a running local LLM in minutes. We also briefly mention alternative tools at the end of the chapter.

If you want to go deeper into Ollama, including tool use, agents, RAG, and advanced configuration, see my book [Ollama in Action](https://leanpub.com/ollama-in-action).


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
| llava:7b | 4.7 GB | Vision model — can analyze images |


## Using Ollama from Python

The **ollama** Python SDK provides a clean interface to the local Ollama service.

```bash
uv pip install ollama
```

### Basic Text Generation

The simplest use of the Ollama SDK — send a prompt and print the response:

```python
# ollama_text.py - Basic text generation with a local model

import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "Briefly explain what a neural network is."}
    ]
)

print(response.message.content)
```

This is similar in structure to the cloud API examples from the previous chapter, but the request never leaves your machine.

### Streaming Responses

For interactive applications, streaming lets users see output as it's generated rather than waiting for the complete response:

```python
# ollama_streaming.py - Streaming responses for real-time output

import ollama

stream = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

for chunk in stream:
    print(chunk.message.content, end="", flush=True)
print()  # final newline
```

Each chunk contains a small piece of the response. The **flush=True** argument ensures text appears immediately rather than being buffered.


## Reasoning with Local Models

Some local models support explicit chain-of-thought reasoning, where the model shows its thinking process before providing a final answer. DeepSeek-R1 is particularly good at this.

First pull the model:

```bash
ollama pull deepseek-r1:7b
```

Here is an example that extracts both the reasoning trace and the final answer:

```python
# ollama_reasoning.py - Chain-of-thought reasoning with DeepSeek-R1

import ollama
import json

def reason_about(question: str, model: str = "deepseek-r1:7b") -> dict:
    """Ask a question and extract both reasoning and final answer."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    content = response.message.content

    # DeepSeek-R1 wraps reasoning in <think>...</think> tags
    reasoning = ""
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

The model's reasoning trace shows each step of its thinking, making the output more transparent and debuggable than a black-box answer. This is especially valuable for math, logic, and planning tasks.


## Conversation Memory with Ollama

Cloud APIs handle conversation history by passing the full message list with each request. With local models the same pattern applies, but since there are no per-token costs, you can maintain longer conversations without worrying about expense.

Here is an example that maintains a conversation with memory across multiple exchanges and uses a system prompt to shape the assistant's personality:

```python
# ollama_memory.py - Conversation with persistent memory

import ollama

class LocalAssistant:
    """A simple conversational assistant that maintains message history."""

    def __init__(self, model: str = "llama3.2:3b", system_prompt: str = ""):
        self.model = model
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining conversation history."""
        self.messages.append({"role": "user", "content": user_message})
        response = ollama.chat(model=self.model, messages=self.messages)
        reply = response.message.content
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

Note that unlike cloud APIs, keeping long conversation histories in local models is free — there are no per-token costs. The main constraint is the model's context window size, which varies by model (typically 4K to 128K tokens).


## Prompt Caching for Performance

When you send the same long context (a document, a knowledge base, or a detailed system prompt) with multiple questions, Ollama can cache the prompt processing to dramatically speed up subsequent requests. This happens automatically when the prefix of the prompt is identical across requests.

Here is an example that demonstrates the speedup:

```python
# ollama_caching.py - Prompt caching benchmark

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

# A long static context that stays the same across queries
CONTEXT = """
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
""" * 20  # repeat to create a substantial context


def timed_query(question: str, label: str) -> float:
    """Send a query with the shared context and measure prompt processing time."""
    payload = {
        "model": MODEL,
        "keep_alive": "60m",  # keep model and cache in memory
        "prompt": f"{CONTEXT}\n\nQuestion: {question}",
        "stream": False,
        "options": {"num_ctx": 4096}
    }
    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload)
    elapsed = time.time() - start
    data = resp.json()

    # prompt_eval_duration is in nanoseconds
    eval_ms = data.get("prompt_eval_duration", 0) / 1_000_000
    print(f"[{label}] Wall time: {elapsed:.2f}s | Prompt eval: {eval_ms:.0f}ms")
    return eval_ms


# First request: cold start, processes the full context
time_a = timed_query("When was Python created?", "Cold start")

# Second request: same context prefix, different question — cache hit
time_b = timed_query("What paradigms does Python support?", "Cache hit")

if time_a > 0 and time_b > 0:
    print(f"\nSpeedup: {time_a / time_b:.1f}x faster on cached prompt")
```

The key settings for prompt caching:

- **keep_alive**: Set to a long duration (e.g., "60m") so the model and its KV cache stay in memory between requests.
- **Identical prefix**: The cached portion must be exactly the same. If even one character of the context changes, the cache is invalidated.
- **Consistent num_ctx**: The context window size must match between requests.

Prompt caching is especially valuable for applications like document Q&A, where you load a long document once and then answer many questions about it.


## OpenAI-Compatible API

Ollama exposes an OpenAI-compatible API endpoint, which means you can use the standard **openai** Python library to talk to local models. This is useful if you want to write code that can switch between cloud and local models by changing only the base URL:

```python
# ollama_openai_compat.py - Using local Ollama with the OpenAI SDK

from openai import OpenAI

# Point the OpenAI client at the local Ollama server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # Ollama doesn't require authentication locally
)

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the difference between a list and a tuple in Python?"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

This compatibility layer means you can prototype with local models and then switch to OpenAI, Gemini, or another provider by changing the client configuration — the rest of your code stays the same.


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

Running LLMs locally with Ollama gives you a private, cost-free, offline-capable alternative to cloud APIs. The setup is straightforward — install Ollama, pull a model, and start making API calls from Python. Features like streaming, conversation memory, prompt caching, and reasoning models make local models practical for many real applications.

The main tradeoff is capability: the largest models that run locally (7-14B parameters on typical hardware) are less capable than frontier cloud models with hundreds of billions of parameters. For many tasks — code assistance, text summarization, data extraction, conversational interfaces — local models perform well enough, and the privacy and cost benefits make them the better choice.
