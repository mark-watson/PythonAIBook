"""
Text summarization using a local Hugging Face model with PyTorch.

Uses the facebook/bart-large-cnn model via AutoModelForSeq2SeqLM.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    model_name = "facebook/bart-large-cnn"
    print(f"Loading summarization model ({model_name})...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    text = (
        "The President sent a request for changing the debt ceiling to "
        "Congress. The president might call a press conference. The Congress "
        "was not oblivious of what the Supreme Court's majority had ruled on "
        "budget matters. Even four Justices had found nothing to criticize in "
        "the President's requirement that the Federal Government's four-year "
        "spending plan. It is unclear whether or not the President and "
        "Congress can come to an agreement before Congress recesses for a "
        "holiday. There is major disagreement between the Democratic and "
        "Republican parties on spending."
    )

    print(f"\nOriginal text ({len(text.split())} words):\n{text}\n")

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        **inputs, max_length=60, num_beams=4, early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"Summary:\n{summary}")


if __name__ == "__main__":
    main()
