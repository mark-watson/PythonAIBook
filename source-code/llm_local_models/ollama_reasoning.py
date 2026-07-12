# ollama_reasoning.py - Chain-of-thought reasoning with DeepSeek-R1
#
# DeepSeek-R1 wraps its internal reasoning process in <think>...</think> tags.
# This script extracts both the reasoning trace and the final answer,
# making the model's thought process transparent and debuggable.
#
# Inspired by the reasoning examples in "Ollama in Action" but uses a
# different problem domain (combinatorics) and a self-contained approach
# without external config dependencies.
#
# Requirements: ollama pull deepseek-r1:7b
# Run: uv run ollama_reasoning.py

import ollama


def reason_about(question: str, model: str = "deepseek-r1:7b") -> dict:
    """Ask a question and extract both reasoning and final answer."""
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": question}]
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
