# ollama_caching.py - Prompt caching benchmark
#
# Demonstrates Ollama's automatic prompt caching. When the same context
# prefix is sent with multiple queries, Ollama reuses the cached KV
# computations from the first request, dramatically speeding up subsequent ones.
#
# Inspired by the prompt_caching examples in "Ollama in Action" but uses a
# self-contained context (no external data files) and a different benchmark
# approach to illustrate the caching mechanism.
#
# Requirements: ollama pull llama3.2:3b
# Run: uv run ollama_caching.py

import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

# A long static context that stays the same across queries
CONTEXT = (
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
    """Send a query with the shared context and measure prompt processing time."""
    payload = {
        "model": MODEL,
        "keep_alive": "60m",  # keep model and cache in memory
        "prompt": f"{CONTEXT}\n\nQuestion: {question}",
        "stream": False,
        "options": {"num_ctx": 4096},
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
