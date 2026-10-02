# ollama_reasoning.py - Chain-of-thought reasoning with DeepSeek-R1
#
# DeepSeek-R1 wraps its internal reasoning process in <think>...</think> tags.
# This script extracts both the reasoning trace and the final answer,
# making the model's thought process transparent and debuggable.
#
# litellm surfaces a reasoning trace as the message's reasoning_content field
# when the provider sends it as a separate field; DeepSeek-R1 served by Ollama
# leaves it inline, so we fall back to splitting the <think> block out of the
# content.
#
# Inspired by the reasoning examples in "Ollama in Action" but uses a
# different problem domain (combinatorics) and a self-contained approach
# without external config dependencies.
#
# Requirements: uv sync; ollama pull deepseek-r1:7b
# Run: uv run python ollama_reasoning.py

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
