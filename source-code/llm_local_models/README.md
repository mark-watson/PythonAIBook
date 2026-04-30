# LLM Local Models – Source Code

This directory contains example code for the **LLMs with Local Models** chapter.

## Prerequisites

Install [Ollama](https://ollama.com):

```bash
brew install ollama    # macOS
ollama serve           # start the service
```

Pull the models used in the examples:

```bash
ollama pull llama3.2:3b
ollama pull deepseek-r1:7b    # for reasoning example
```

## Examples

- **ollama_text.py** — Basic text generation with a local model
- **ollama_streaming.py** — Streaming responses for real-time output
- **ollama_reasoning.py** — Chain-of-thought reasoning with DeepSeek-R1
- **ollama_memory.py** — Multi-turn conversation with history
- **ollama_caching.py** — Prompt caching benchmark (cold vs warm start)
- **ollama_openai_compat.py** — Using the OpenAI SDK with local Ollama

## Running

All examples use `uv run`:

```bash
uv run ollama_text.py
uv run ollama_streaming.py
uv run ollama_reasoning.py
uv run ollama_memory.py
uv run ollama_caching.py
uv run ollama_openai_compat.py
```
