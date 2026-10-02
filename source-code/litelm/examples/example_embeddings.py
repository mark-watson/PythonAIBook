"""Embeddings through the uniform interface.

Every provider registered by default exposes ``/embeddings`` on its
OpenAI-compatible base URL, so ``litelm.embedding`` covers OpenAI and a local
Ollama embedding model alike (``ollama pull nomic-embed-text``).

Run::

    OPENAI_API_KEY=... uv run example_embeddings.py
    LITELM_EMBED_MODEL=ollama/nomic-embed-text uv run example_embeddings.py
"""

import math
import os

import litelm

MODEL = os.environ.get("LITELM_EMBED_MODEL", "openai/text-embedding-3-small")
SENTENCES = [
    "A neural network learns weights from examples.",
    "Gradient descent updates the weights of a model.",
    "I would like a large pepperoni pizza.",
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def main() -> None:
    vectors = litelm.embedding(MODEL, SENTENCES)
    print(f"model: {MODEL}")
    print(f"{len(vectors)} vectors of {len(vectors[0])} dimensions\n")

    # Two sentences about training should sit closer together than either does
    # to the pizza order.
    print(f"related    (0, 1): {cosine(vectors[0], vectors[1]):.4f}")
    print(f"unrelated  (0, 2): {cosine(vectors[0], vectors[2]):.4f}")
    print(f"unrelated  (1, 2): {cosine(vectors[1], vectors[2]):.4f}")


if __name__ == "__main__":
    main()
