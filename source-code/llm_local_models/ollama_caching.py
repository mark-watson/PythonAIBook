# ollama_caching.py - Prompt caching benchmark
#
# Demonstrates Ollama's automatic prompt caching. When the same context
# prefix is sent with multiple queries, Ollama reuses the cached KV
# computations from the first request, so the second request only has to
# evaluate the handful of new tokens in the question.
#
# This demo uses Ollama's own Python SDK rather than litellm, because the
# measurement depends on fields of the native response: prompt_eval_duration
# and prompt_eval_count. litellm normalizes responses into the OpenAI shape and
# does not carry Ollama's cache timing metrics, so this is the one place where
# the native client earns its keep.
#
# Inspired by the prompt_caching examples in "Ollama in Action" but uses a
# self-contained context (no external data files) and a different benchmark
# approach to illustrate the caching mechanism.
#
# Requirements: uv sync; ollama pull llama3.2:3b
# Run: uv run python ollama_caching.py

import secrets

import ollama

MODEL = "llama3.2:3b"

# keep_alive holds the model (and therefore the cached prompt prefix) in memory
# between requests; num_ctx pins the context window so both runs match.
KEEP_ALIVE = "60m"
OPTIONS = {"num_ctx": 4096}

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
    """Send a query with the shared context; report the prompt-eval time."""
    response = ollama.generate(
        model=MODEL,
        prompt=f"{CONTEXT}\n\nQuestion: {question}",
        keep_alive=KEEP_ALIVE,
        options=OPTIONS,
    )
    assert isinstance(response, ollama.GenerateResponse), (
        "Expected a non-streaming response"
    )

    # prompt_eval_duration is in nanoseconds and covers only the prompt tokens
    # Ollama actually had to evaluate, which is exactly what caching reduces.
    eval_ms = (response.prompt_eval_duration or 0) / 1_000_000
    print(
        f"[{label}] Prompt eval: {eval_ms:.0f}ms | "
        f"prompt tokens: {response.prompt_eval_count}"
    )
    return eval_ms


# First request: cold start, processes the full context
time_a = timed_query("When was Python created?", "Cold start")

# Second request: same context prefix, different question — cache hit
time_b = timed_query("What paradigms does Python support?", "Cache hit")

if time_a > 0 and time_b > 0:
    print(f"\nSpeedup: {time_a / time_b:.1f}x faster on cached prompt")
