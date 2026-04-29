"""
Zero-shot text classification using a local Hugging Face model with PyTorch.

Uses the MoritzLaurer/deberta-v3-base-zeroshot-v2.0 model.
"""

from pprint import pprint
from transformers import pipeline


def main():
    print("Loading zero-shot classification model...")
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    )

    text = (
        "Hi, I recently bought a device from your company but it is not "
        "working as advertised and I would like to get reimbursed!"
    )
    candidate_labels = ["refund", "faq", "legal"]

    print(f"\nInput text: {text}")
    print(f"Candidate labels: {candidate_labels}\n")

    result = classifier(text, candidate_labels)
    pprint(result)


if __name__ == "__main__":
    main()
