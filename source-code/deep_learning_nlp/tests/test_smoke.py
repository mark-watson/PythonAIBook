"""Import-only smoke tests.

All three scripts guard their real work behind `if __name__ == "__main__":`,
so importing them here does not download the huge HuggingFace models
(sentence-transformers, BART, DeBERTa-v3 zero-shot). This just proves each
module parses and its deps resolve.
"""


def test_sentence_similarity_imports() -> None:
    import sentence_similarity

    assert callable(sentence_similarity.main)


def test_summarization_imports() -> None:
    import summarization

    assert callable(summarization.main)


def test_zero_shot_classification_imports() -> None:
    import zero_shot_classification

    assert callable(zero_shot_classification.main)
