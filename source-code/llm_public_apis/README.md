# LLM Public APIs – Source Code

This directory contains example code for the **LLMs with Public APIs** chapter.

## Google Gemini Examples

Requires a Google AI API key ([get one here](https://aistudio.google.com/apikey)):

```bash
export GOOGLE_API_KEY="your-api-key"
uv pip install google-genai Pillow
```

- **gemini_text.py** — Basic text generation
- **gemini_temperature.py** — Effect of temperature on output creativity
- **gemini_thinking.py** — Extended reasoning with thinking budget
- **gemini_conversation.py** — Multi-turn conversation with history
- **gemini_image.py** — Multimodal image analysis (requires a `photo.jpg`)
- **gemini_structured.py** — Extracting structured JSON from text

## OpenAI Examples

Requires an OpenAI API key ([get one here](https://platform.openai.com/api-keys)):

```bash
export OPENAI_API_KEY="your-api-key"
uv pip install openai
```

- **openai_text.py** — Basic text generation with GPT-5.4-nano
- **openai_search.py** — Web search augmented generation

## Fireworks.ai Examples

Requires a Fireworks API key ([get one here](https://fireworks.ai/api-keys)):

```bash
export FIREWORKS_API_KEY="your-api-key"
uv pip install openai
```

Fireworks uses an OpenAI-compatible API, so we use the `openai` SDK with a custom base URL.
The default model is `deepseek-v4-flash`.

- **fireworks_text.py** — Basic text generation
- **fireworks_temperature.py** — Effect of temperature on output creativity
- **fireworks_thinking.py** — Extended reasoning with DeepSeek thinking mode
- **fireworks_conversation.py** — Multi-turn conversation with history
- **fireworks_structured.py** — Extracting structured JSON from text

## Architecture

![Public cloud LLM API architecture for Google Gemini and OpenAI](FIG_llm_public_apis.jpg)
