# Natural Language Processing Using Deep Learning

I spent several years in the 1980s using symbolic AI approaches to Natural Language Processing (NLP) like augmented transition networks and conceptual dependency theory with mixed results. For small vocabularies and small domains of discourse these techniques yielded modestly successful results. I now only use Deep Learning approaches to NLP in my work.

Deep Learning in NLP is a branch of machine learning that utilizes deep neural networks to understand, interpret and generate human language. It has revolutionized the field of NLP by improving the accuracy of various NLP tasks such as text classification, language translation, sentiment analysis, and natural language generation (e.g., ChatGPT).

Deep learning models such as Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), and Transformer models have been used to achieve state-of-the-art performance on various NLP tasks. These models have been trained on large amounts of text data, which has allowed them to learn complex patterns in human language and improve their understanding of the context and meaning of words.

The use of pre-trained models, such as BERT and GPT-4, has also become popular in NLP and I use both frequently for my work. These models have been pre-trained on a large corpus of text data, and can be fine-tuned for a specific task, which significantly reduces the amount of data and computing resources required to train a derived model.

Deep learning in NLP has been applied in various industries such as chatbots, automated customer service, and language translation services. It has also been used in research areas such as natural language understanding, question answering, and text summarization.

In the last decade deep learning techniques have solved most NLP problems, at least in a "good enough" engineering sense. In this chapter we will experiment with a few useful pre-trained models that you can run locally on your laptop using PyTorch.

{width: "80%"}
![Architecture diagram for the Deep Learning NLP example](FIG_deep_learning_nlp.jpg)

## Hugging Face and the Transformers Library

Hugging Face provides an extensive library of pre-trained models and the **transformers** Python library that allows developers to quickly integrate NLP capabilities into their applications. The pre-trained models are based on state-of-the-art transformer architectures, trained on large corpora of data. Hugging Face maintains a task page listing all kinds of machine learning that they support at [https://huggingface.co/tasks](https://huggingface.co/tasks) for task domains:

- Computer Vision
- Natural Language Processing
- Audio
- Tabular Data
- Multimodal
- Reinforcement Learning

As an open source and open model company, Hugging Face is a leading provider of NLP technology. They have developed a library of pre-trained models, including models based on transformer architectures such as BERT, DeBERTa, and GPT variants, which can be fine-tuned for various tasks such as language understanding, language translation, and text generation.

The requirements for the examples in this chapter are:

```bash
uv pip install torch transformers sentence-transformers
```

The examples for this chapter are in the directory **source-code/deep_learning_nlp**.

All models will be downloaded automatically to **~/.cache/huggingface** the first time you run each script. Subsequent runs will use the cached models without re-downloading.

### Summarizing Text Using a Pre-trained Model on Your Laptop

We use the **facebook/bart-large-cnn** model for text summarization. This model runs locally on your laptop using PyTorch — no API keys or cloud services needed. The downloaded model and associated files are a little less than two gigabytes of data.

In the current version of the transformers library (v5+), we load the model and tokenizer explicitly using `AutoModelForSeq2SeqLM` and `AutoTokenizer`:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/bart-large-cnn"
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

inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(
    **inputs, max_length=60, num_beams=4, early_stopping=True
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

Here is the output from running **summarization.py**:

```bash
$ python summarization.py
Loading summarization model (facebook/bart-large-cnn)...

Original text (87 words):
The President sent a request for changing the debt ceiling to Congress...

Summary:
The President sent a request for changing the debt ceiling to
Congress. The Congress was not oblivious of what the Supreme
Court's majority had ruled on budget matters. Even four Justices
had found nothing to criticize in the President's requirement
that the Federal Government's four-year spending plan be changed
```

Note: for production summarization work with longer documents, consider using larger instruction-tuned models (e.g., Llama 3 or Phi-4) that support context windows of 128K+ tokens. The BART model used here is limited to 1,024 tokens of input but is excellent for learning and prototyping.

### Zero Shot Classification Using a Local Model

Zero shot classification models work by specifying which classification labels you want to assign to input texts — no pre-training on labeled examples is required. We use the **MoritzLaurer/deberta-v3-base-zeroshot-v2.0** model, which is a modern and efficient replacement for the older **bart-large-mnli** model:

```python
from pprint import pprint
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
)

text = (
    "Hi, I recently bought a device from your company but it is not "
    "working as advertised and I would like to get reimbursed!"
)
candidate_labels = ["refund", "faq", "legal"]

result = classifier(text, candidate_labels)
pprint(result)
```

Here is the output from running **zero_shot_classification.py**:

```bash
$ python zero_shot_classification.py
Loading zero-shot classification model...

Input text: Hi, I recently bought a device from your company but
it is not working as advertised and I would like to get reimbursed!
Candidate labels: ['refund', 'faq', 'legal']

{'labels': ['refund', 'legal', 'faq'],
 'scores': [0.9839024543762207, 0.01550459023565054, 0.000593014934565872],
 'sequence': 'Hi, I recently bought a device from your company but it is not '
             'working as advertised and I would like to get reimbursed!'}
```

The model correctly classifies the customer message as a "refund" request with 98.4% confidence. This approach is powerful because you can define any set of labels at runtime without training a custom model.

## Comparing Sentences for Similarity Using Transformer Models

The [sentence-transformers](https://www.sbert.net/) library provides an easy-to-use interface for computing sentence embeddings using PyTorch. We use the **all-MiniLM-L6-v2** model which is small (80 MB) and fast while producing high-quality embeddings:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The IRS has new tax laws.",
    "Congress debating the economy.",
    "The politician fled to South America.",
    "Canada and the US will be in the playoffs.",
    "The cat ran up the tree.",
    "The meal tasted good but was expensive and perhaps not worth the price.",
]

# Encode all sentences
embeddings = model.encode(sentences)

# Compute cosine similarities between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)

# Collect and sort all unique pairs
pairs = []
for i in range(len(cos_sim) - 1):
    for j in range(i + 1, len(cos_sim)):
        pairs.append((cos_sim[i][j].item(), i, j))

pairs.sort(key=lambda x: x[0], reverse=True)

print("Top-8 most similar pairs:")
for score, i, j in pairs[:8]:
    print(f"  {score:.4f}  {sentences[i]}")
    print(f"          {sentences[j]}")
```

Output from running **sentence_similarity.py**:

```bash
$ python sentence_similarity.py
Loading sentence-transformers model (all-MiniLM-L6-v2)...

Top-8 most similar pairs:
  0.1793  The IRS has new tax laws.
          Congress debating the economy.

  0.1210  Congress debating the economy.
          Canada and the US will be in the playoffs.

  0.1131  Congress debating the economy.
          The meal tasted good but was expensive and perhaps not worth the price.

  0.1061  Congress debating the economy.
          The politician fled to South America.

  0.0901  The politician fled to South America.
          The meal tasted good but was expensive and perhaps not worth the price.

  0.0677  The politician fled to South America.
          The cat ran up the tree.

  0.0432  The cat ran up the tree.
          The meal tasted good but was expensive and perhaps not worth the price.

  0.0388  Congress debating the economy.
          The cat ran up the tree.
```

The results make intuitive sense: sentences about government and economics cluster together, while unrelated sentences about cats and meals have very low similarity scores.

A common use case is a customer service chatbot where we match the user's question with all recorded questions that have accepted "canned answers." The runtime to get the best match is **O(N)** where **N** is the number of previously recorded user questions. The cosine similarity calculation, given two embedding vectors, is very fast.

For a practical application you can use the `cos_sim` function from the sentence-transformers utility library:

```python
>>> from sentence_transformers import util
>>> util.cos_sim(embeddings[0], embeddings[1])
tensor([[0.1793]])
```

## Deep Learning Natural Language Processing Wrap-up

In this chapter we have seen examples of how effective deep learning is for NLP using PyTorch and the Hugging Face transformers ecosystem. All three examples — summarization, zero-shot classification, and sentence similarity — run locally on your laptop without requiring API keys or cloud services.

I worked on other methods of NLP over a 25-year period and I ask you, dear reader, to take my word on this: deep learning has revolutionized NLP and for almost all practical NLP applications, deep learning libraries and pre-trained models from organizations like Hugging Face should be the first thing that you consider using.

## Optional Practice Problems

Here are optional practice problems designed to help you apply the concepts from this chapter. They build directly on the existing code examples in `source-code/deep_learning_nlp`.

### 1. Zero-Shot Classifier with Confidence Thresholding (Easy)
Extend the script [zero_shot_classification.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/deep_learning_nlp/zero_shot_classification.py) to handle realistic customer service routing.
- **Task:** Update the candidate labels to classify incoming messages into product categories: `"software"`, `"hardware"`, `"billing"`, and `"shipping"`.
- **Requirements:**
  1. Test your classifier on at least three different messages representing these classes (e.g., "The latest update crashes on launch", "I was charged twice on my credit card", "When will my package arrive?").
  2. Implement a threshold check. If the top classification score is below `0.6`, print a warning message: `"Low confidence classification. Routing to a human representative."` Otherwise, print the winning category and its score.

### 2. Semantic FAQ Search Engine (Medium)
Build a simple, interactive FAQ matching engine by extending [sentence_similarity.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/deep_learning_nlp/sentence_similarity.py).
- **Task:** Write a Python function that accepts a user question, finds the closest matching question from a pre-defined FAQ database, and prints the corresponding answer.
- **Data:** Use a dictionary mapping common questions to answers:
  ```python
  faq_database = {
      "How do I return a defective product?": "Please request a return label via our website and send it back to our warehouse.",
      "What payment methods do you accept?": "We accept credit cards (Visa, MasterCard, Amex) and PayPal.",
      "Where is my package?": "You can find your order tracking number in your shipping confirmation email."
  }
  ```
- **Requirements:**
  1. Embed the user's search query and compute its cosine similarity against the embedded FAQ questions.
  2. Retrieve the question with the highest similarity.
  3. If the highest similarity score is above `0.5`, print the matching question and its answer.
  4. If it is below `0.5`, print: `"Sorry, we couldn't find a direct answer. Would you like to rephrase?"`

### 3. Long Document Summarization Pipeline (Hard)
The `facebook/bart-large-cnn` model has a strict context limit of 1,024 tokens. If you feed it a large text document, it will fail or truncate the input.
- **Task:** Build a modular summarization pipeline in [summarization.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/deep_learning_nlp/summarization.py) that can handle arbitrarily long text by chunking.
- **Requirements:**
  1. Find or write a long text block of 1,500 to 2,000 words (e.g., a news article or short story).
  2. Write a function that splits this text into clean chunks of at most 800 tokens each (ensuring each chunk is clean and doesn't cut mid-sentence).
  3. Loop over the chunks, summarize each chunk, and collect the summaries.
  4. Finally, concatenate the individual summaries and pass the combined text back through the BART model to generate a single, cohesive executive summary.
