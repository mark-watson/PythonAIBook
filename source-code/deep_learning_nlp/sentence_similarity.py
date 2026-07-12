"""
Sentence similarity using the sentence-transformers
library (PyTorch).

Computes cosine similarity between all pairs of sentences
and ranks them by similarity.
"""

from sentence_transformers import (
    SentenceTransformer,
    util,
)


def main():
    print("Loading sentence-transformers model")
    print("  (all-MiniLM-L6-v2)...\n")

    # Load a lightweight sentence embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Sentences to compare for semantic similarity
    sentences = [
        "The IRS has new tax laws.",
        "Congress debating the economy.",
        "The politician fled to South America.",
        "Canada and the US will be in the playoffs.",
        "The cat ran up the tree.",
        "The meal tasted good but was expensive and perhaps not worth the price.",
    ]

    # Encode all sentences into embedding vectors
    embeddings = model.encode(sentences)

    # Compute pairwise cosine similarity matrix
    cos_sim = util.cos_sim(embeddings, embeddings)

    # Collect all unique sentence pairs with scores
    pairs = []
    for i in range(len(cos_sim) - 1):
        for j in range(i + 1, len(cos_sim)):
            # Extract float score from the tensor
            pairs.append((cos_sim[i][j].item(), i, j))

    # Sort by highest similarity first
    pairs.sort(key=lambda x: x[0], reverse=True)

    print("\nTop-8 most similar pairs:")
    for score, i, j in pairs[:8]:
        print(f"  {score:.4f}  {sentences[i]}")
        print(f"          {sentences[j]}")
        print()


if __name__ == "__main__":
    main()
