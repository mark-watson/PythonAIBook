# Part 3 - Large Language Models

Large Language Models (LLMs) are arguably the most transformative development in AI in the last decade. Built on the transformer architecture, LLMs have moved from research curiosities to indispensable tools used daily by millions of people for writing, coding, analysis, and creative work.

We start with an introduction to the transformer architecture, self-attention, tokenization, and the training techniques that produce modern LLMs. We then move to practical chapters: using LLMs through cloud APIs from Google Gemini and OpenAI, and running open-weights models locally on your own hardware with Ollama.

Both practical chapters drive every model through **[litellm](https://github.com/BerriAI/litellm)**, the open-source library that puts a single OpenAI-format interface on more than 100 providers. With litellm the provider is just a prefix in the model name — `gemini/gemini-3-flash-preview`, `openai/gpt-5.4-nano`, `fireworks_ai/accounts/fireworks/models/deepseek-v4p1-flash`, `nvidia_nim/meta/llama-3.1-8b-instruct`, `ollama_chat/llama3.2:3b` — so the same Python code reaches a cloud API or a model running on your own machine by changing one string.