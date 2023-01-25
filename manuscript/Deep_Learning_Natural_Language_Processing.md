# Natural Language Processing

I spent several years in the 1980s using symbolic AI approaches to Natural Language Processing (NLP) like augmented transition networks and conceptual dependency theory with mixed results. For small vocabularies and small domains of discourse these techniques yielded modestly successful results. I now only use Deep Learning approaches to NLP in my work.

Deep Learning in NLP is a branch of machine learning that utilizes deep neural networks to understand, interpret and generate human language. It has revolutionized the field of NLP by improving the accuracy of various NLP tasks such as text classification, language translation, sentiment analysis, and natural language generation (e.g., ChatGPT).

Deep learning models such as Recurrent Neural Networks (RNNs), Convolutional Neural Networks (CNNs), and Transformer models have been used to achieve state-of-the-art performance on various NLP tasks. These models have been trained on large amounts of text data, which has allowed them to learn complex patterns in human language and improve their understanding of the context and meaning of words.

The use of pre-trained models, such as BERT and GPT-3, has also become popular in NLP and I use both frequently for my work. These models have been pre-trained on a large corpus of text data, and can be fine-tuned for a specific task, which significantly reduces the amount of data and computing resources required to train a derived model.

Deep learning in NLP has been applied in various industries such as chatbots, automated customer service, and language translation services. It has also been used in research areas such as natural language understanding, question answering, and text summarization.

So, Deep Learning in NLP has greatly improved the performance of various NLP tasks by utilizing deep neural networks to understand and interpret human language. The use of pre-trained models has also made it easier to fine-tune models for specific tasks, which has led to a wide range of applications in industry and research.

In the last decade deep learning techniques have solved most NLP problems, at least in a "good enough" engineering sense. As I write this the ChatGPT model has scored 80% accuracy on the verbal SAT college admissions test. In this chapter we will experiment with a few useful public models that can be used as paid for API calls or in many cases you can run the models yourself.

## OpenAI GPT-3 APIs

OpenAI GPT-3 (Generative Pre-trained Transformer 3) is an advanced language processing model developed by OpenAI. There are three general classes of OpenAI API services:

- GPT-3 which performs a variety of natural language tasks.
- Codex which translates natural language to code.
- DALL·E which creates and edits original images.

GPT-3 is capable of generating human-like text, completing tasks such as language translation, summarization, and question answering, and much more. OpenAI offers GPT-3 APIs, which allow developers to easily integrate GPT-3's capabilities into their applications.

The GPT-3 API provides a simple and flexible interface for developers to access GPT-3's capabilities such as text completion, language translation, and text generation. The API can be accessed using a simple API call, and can be integrated into a wide range of applications such as chatbots, language translation services, and text summarization.

Additionally, OpenAI provides a number of pre-built models that developers can use, such as the GPT-3 language model, the GPT-3 translation model, and the GPT-3 summarization model. These pre-built models allow developers to quickly and easily access GPT-3's capabilities without the need to train their own models.

Overall, the OpenAI GPT-3 APIs provide a powerful and easy-to-use tool for developers to integrate advanced language processing capabilities into their applications, and can be a game changer for developers looking to add natural language processing capabilities to their projects.

We will only use the GPT-3 APIs here. The following examples are derived from the official set of cookbook examples at [https://github.com/openai/openai-cookbook](https://github.com/openai/openai-cookbook). The first example calls the OpenAI GPT-3 Completion API with a sample of input text and the model completes the text (deep-learning/openai/openai-example.py):

{caption: "OpenAI GPT-3 Completion Example"}
{format: python}
![](openai-example.py)

Everytime you run this example you get different output. Here is one example run:

```
$ python openai-example.py
 bread, butter, and tomatoes. She ran into some old friends, but she
```

### Using GPT-3 to Name K-Means Clusters

I get a lot of enjoyment finding simple application examples that solve problems that I had previously spent a lot of time solving with other techniques. As an example, around 2010 a customer and I created some ad hoc ways to name K-Means clusters with meaningful cluster names. Several years later at Capital One, my team brainstormed other techniques for assigning meaningful cluster names for the patent [SYSTEM TO LABEL K-MEANS CLUSTERS WITH HUMAN UNDERSTANDABLE LABELS](https://patents.justia.com/patent/20210357429).

One of the OpenAI example Jupyter notebooks [https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb) solves this problem elegantly using a text prompt like:

```
f'What do the following customer reviews have in common?\n\nCustomer reviews:\n"""\n{reviews}\n"""\n\nTheme:'
```

where the variable **reviews** contains the concatenated recipe reviews in a specific cluster. The recipe clusters are named like:

```
Cluster 0 Theme:  All of the reviews are positive and the customers are satisfied with the product they purchased.

Cluster 1 Theme:  All of the reviews are about pet food.

Cluster 2 Theme:  All of the reviews are positive and express satisfaction with the product.

Cluster 3 Theme:  All of the reviews are about food or drink products.
```

### Using GPT-3 to Translate Natural Language Queries to SQL

Another example of a long term project I had that is now easily solved with the OpenAI GPT-3 models is translating natural language queries to SQL queries. I had an example I wrote for the first two editions of my [Java AI book](https://leanpub.com/javaai) (I later removed this example because the code was difficult to follow). I later reworked this example in Common Lisp and used both versions in several consulting projects in the late 1990s and early 2000s.

I refer you to one of the official OpenAI examples [https://github.com/openai/openai-cookbook/blob/main/examples/Backtranslation_of_SQL_queries.py](https://github.com/openai/openai-cookbook/blob/main/examples/Backtranslation_of_SQL_queries.py). In my Java and Common Lisp NLP query examples, I would test generated SQL queries against a database to ensure they were legal queries, etc. and if you modify OpenAI's example I suggest you do the same.

Here is the output to OpenAI's example:

```sql
$ python Backtranslation_of_SQL_queries.py
SELECT department.name
FROM department
JOIN employee ON department.id = employee.department_id
JOIN salary_payments ON employee.id = salary_payments.employee_id
WHERE salary_payments.date BETWEEN '2021-06-01' AND '2021-06-30'
GROUP BY department.name
HAVING COUNT(*) > 10
```


## Hugging Face APIs

Hugging Face provides an extensive library of pre-trained models and a set of easy-to-use APIs that allow developers to quickly and easily integrate NLP capabilities into their applications. The pre-trained models are based on the state-of-the-art transformer architectures, which have been trained on large corpus of data and can be fine-tuned for specific tasks, making it easy for developers to add NLP capabilities to their projects. Hugging Face maintains a task page listing all kinds of machine learning that they support [https://huggingface.co/tasks](https://huggingface.co/tasks) for task domains:

- Computer Vision
- Natural Language Processing
- Audio
- Tabular Data
- Multimodal
- Reinforcement Learning

As a open source and open model company, Hugging Face is a provider of NLP technology, with a focus on developing and providing state-of-the-art pre-trained models and tools for NLP tasks. They have developed a library of pre-trained models, including models based on transformer architectures such as BERT, GPT-2, and GPT-3, which can be fine-tuned for various tasks such as language understanding, language translation, and text generation.

Hugging Face also provides a set of APIs, which allows developers to easily access the functionality of these pre-trained models. The APIs provide a simple and flexible interface for developers to access the functionality of these models, such as text completion, language translation, and text generation. This allows developers to quickly and easily integrate NLP capabilities into their applications, without the need for extensive knowledge of NLP or deep learning.

The Hugging Face APIs are available via a simple API call and are accessible via an API key. They support a wide range of programming languages such as Python, Java and JavaScript, making it easy for developers to integrate them into their application.

Since my personal interests are mostly in Natural Language Processing (NLP) as used for processing text data, automatic extraction of structured data from text, and question answering systems, I will just list their NLP task types:

![](hfnlp.png)

### Conversation Agent

The fist example using the Hugging Face conversation model. The example is in the file **deep-learning/huggingface_apis/hf-conversation.py**:

{caption: "Hugging Face Conversation Example"}
{format: python}
![](hf-conversation.py)

You will get different results everytime you run this script. Here is one example:

```json
$ python hf-conversation.py 
{'conversation': {'generated_responses': ["It's Die Hard for sure.",
                                          "It's the best movie ever."],
                  'past_user_inputs': ['Which movie is the best ?',
                                       'Can you explain why ?']},
 'generated_text': "It's the best movie ever."
}
```


### Coreference: Resolve Pronouns to Proper Nouns in Text

{caption: "Hugging Face Conversation Example"}
{format: python}
![](hf-coreference.py)

Here is example output (I am only showing the highest scored results for each query):

```
$ python hf-coreference.py 
[{'score': 0.16963981091976166,
  'sequence': 'the answer to the universe is no.',
  'token': 2053,
  'token_str': 'no'},
 {'score': 0.07344783842563629,
  'sequence': 'the answer to the universe is nothing.',
  'token': 2498,
  'token_str': 'nothing'}]
[{'score': 0.9037206768989563,
  'sequence': 'john smith bought a car. he drives it fast.',
  'token': 2002,
  'token_str': 'he'},
 {'score': 0.015135547146201134,
  'sequence': 'john smith bought a car. john drives it fast.',
  'token': 2198,
  'token_str': 'john'}]
```

### GPT-2 Example

{caption: "Using GPT-2 Hosted as a Hugging Face API"}
{format: python}
![](hf-gpt2_test.py)

Here is example output:

```
$ python hf-gpt2_test.py 
[{'generated_text': 'Can you please let us know more details about your '
                    'iphone?\n'
                    '\n'
                    'If you purchased an iPhone to get around Wi-Fi, such as '
                    "using your iPhone's Wi-Fi or Bluetooth 3.1 to get around "
                    'Wi-Fi, the'}]
```

### Answer Questions From Text

{caption: "Read Text and Answer Questions"}
{format: python}
![](hf-qa.py)

Example output:

```
$ python hf-qa.py
{'error': 'overloaded'}

```

### Calculate Semantic Similarity of Sentences

Given a list of sentences we can calculate sentence embeddings for each sentence. Any new sentence and be matched by calculating its embedding and finding the closest cosine similarity match. Contents of file **hf-sentence_similarities.py**:


{caption: "Sentence Similarity"}
{format: python}
![](hf-sentence_similarities.py)

Here is example output:

```
$ python hf-sentence_similarities.py 
[0.6945773363113403, 0.9429150819778442, 0.2568760812282562]
```

Here we are using the one of the free Hugging Face APIs. At the end of this chapter we will use an alternative sentence embedding model that you can easily run on your laptop.

### Summarizing Text.  NOT USE?

{caption: "Hugging Model for Summazing Text"}
{format: python}
![](hf-summarization.py)

Here is some sample output:

```
 $ python hf-summarization.py 
[{'summary_text': 'The President went to Congress. The Congress was not '
                  "oblivious of what the Supreme Court's majority had ruled. "
                  'Even four Justices had found nothing to criticize in the '
                  "President's requirement that the Federal Government's "
                  'four-year term be extended. The President went back to '
                  'Congress, and the Congress agreed.'}]
```


### Zero Shot Classification. NOT USE?

{caption: "Hugging Model for Zero Shot Classification"}
{format: python}
![](hf-zero_shot_classification.py)

Here is some example output:

```
$ python hf-zero_shot_classification.py 
{'labels': ['refund', 'faq', 'legal'],
 'scores': [0.877787709236145, 0.10522633045911789, 0.01698593981564045],
 'sequence': 'Hi, I recently bought a device from your company but it is not '
             'working as advertised and I would like to get reimbursed!'}
```

## Comparing Sentences for Similarity Using Transformer Models

Although I usually use OpenAI and HuggingFace for most of the pre-trained NLP models I use, I recently used a sentence similarity Transformer model from the [Ubiquitous Knowledge Processing Lab](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) for a quick work project and their library support for finding similar sentences is simple to use written with PyTorch. Here is one of their examples, slightly modified for this book:

```python
# pip install sentence_transformers
# The first time this script is run, the sentence_transformers library will
# download a pre-trained model.

# This example is derived from examples at https://www.sbert.net/docs/quickstart.html

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ['The IRS has new tax laws.',
             'Congress debating the economy.',
             'The polition fled to South America.',
             'Canada and the US will be in the playoffs.',
             'The cat ran up the tree',
             'The meal tasted good but was expensive and perhaps not worth the price.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(sentence_embeddings, sentence_embeddings)

#Add all pairs to a list with their cosine similarity score
all_sentence_combinations = []
for i in range(len(cos_sim)-1):
    for j in range(i+1, len(cos_sim)):
        all_sentence_combinations.append([cos_sim[i][j], i, j])

#Sort list by the highest cosine similarity score
all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: x[0], reverse=True)

print("Top-8 most similar pairs:")
for score, i, j in all_sentence_combinations[0:8]:
    print("{} \t {} \t {:.4f}".format(sentences[i],
                                      sentences[j],
                                      cos_sim[i][j]))
```

The output is:

```
$ python sentence_transformer.py
Top-8 most similar pairs:
The IRS has new tax laws. 	 Congress debating the economy. 	 0.1793
Congress debating the economy. 	 Canada and the US will be in the playoffs. 	 0.1210
Congress debating the economy. 	 The meal tasted good but was expensive and perhaps not worth the price. 	 0.1131
Congress debating the economy. 	 The polition fled to South America. 	 0.0963
The polition fled to South America. 	 Canada and the US will be in the playoffs. 	 0.0854
The polition fled to South America. 	 The meal tasted good but was expensive and perhaps not worth the price. 	 0.0826
The polition fled to South America. 	 The cat ran up the tree 	 0.0809
Congress debating the economy. 	 The cat ran up the tree 	 0.0496
```

A common use case might be a customer service facing chatbot where we simply match the user's question with all recorded user questions that have accepted "canned answers." The runtime to get the best match is **O(N)** where **N** is the number of previously recorded user questions. The cosine similarity calculation, given two embedding vectors, is very fast.

In this example we used the [Sentence Transformer utility library util.py](https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py) to calculate the cosine similarities between all combinations of sentence embeddings. For a practical application you can use the **cos_sim** function in **util.py**:

```python
>>> from util
>>> util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])
tensor([[0.1793]])
>>> 
```
