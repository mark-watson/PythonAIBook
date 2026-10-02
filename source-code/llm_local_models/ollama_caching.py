# ollama_caching.py - Prompt caching benchmark
#
# Demonstrates Ollama's automatic prompt caching. When the same context
# prefix is sent with multiple queries, Ollama reuses the cached KV
# computations from the first request, so the second prompt is almost
# entirely a cache hit.
#
# Ollama's native API reports prompt_eval_duration and cached-prompt tokens,
# but litellm's response usage does not surface them, so the hit is measured
# here in wall-clock time: the warm request reuses the cached prefix and
# therefore returns noticeably faster than the cold one.
#
# Inspired by the prompt_caching examples in "Ollama in Action" but uses a
# self-contained context (no external data files) and a different benchmark
# approach to illustrate the caching mechanism.
#
# Requirements: uv sync; ollama pull llama3.2:3b
# Run: uv run python ollama_caching.py

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
