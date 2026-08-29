# Introduction to Transformers and Large Language Models

The transformer architecture, introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al., fundamentally changed artificial intelligence. Before transformers, recurrent neural networks (RNNs) and their variants (LSTMs, GRUs) were the dominant approach for processing sequential data like text. These models processed tokens one at a time, which made them slow to train and poor at capturing relationships between words that were far apart in a sentence.

Transformers replaced this sequential processing with a mechanism called self-attention that allows every token in a sequence to attend to every other token simultaneously. This parallel processing made transformers dramatically faster to train on modern GPU hardware, and the attention mechanism proved remarkably effective at capturing the complex patterns in language. Within a few years, transformer-based models achieved state-of-the-art results across virtually every natural language processing benchmark, and the architecture has since been successfully applied to computer vision, protein folding, music generation, and many other domains.

Large Language Models (LLMs) are transformer models trained on vast corpora of text, often trillions of tokens scraped from the internet, books, and code repositories. Through this training, LLMs develop an impressive ability to generate coherent text, answer questions, write code, reason about problems, and follow complex instructions. Models like Google's Gemini, Anthropic's Claude, OpenAI's GPT series, and Meta's Llama family have become indispensable tools for developers and knowledge workers.

This chapter covers the core concepts behind transformers and LLMs. The practical work of using these models (calling cloud APIs and running models locally) is covered in the next two chapters.


## The Transformer Architecture

The transformer architecture consists of two main components: an **encoder** that reads and processes an input sequence, and a **decoder** that generates an output sequence. Some models use both (the original transformer for translation), while modern LLMs typically use only the decoder (GPT, Claude, Gemini, Llama) or only the encoder (BERT, for understanding tasks).

### Self-Attention: The Core Innovation

The key insight of the transformer is the self-attention mechanism. For each token in a sequence, self-attention computes how much that token should "attend to" every other token. This is done by projecting each token's embedding into three vectors:

- **Query (Q)**: What this token is looking for
- **Key (K)**: What this token offers to other tokens
- **Value (V)**: The actual information this token contributes

The attention score between two tokens is computed as the dot product of the query of one token with the key of another, scaled by the square root of the dimension. These scores are passed through a softmax function to produce weights, which are then used to compute a weighted sum of the value vectors. The result is that each token's representation becomes a context-aware blend of information from all tokens in the sequence.

The mathematical formulation is:

    Attention(Q, K, V) = softmax(QK^T / √d_k) V

where d_k is the dimension of the key vectors. The scaling factor √d_k prevents the dot products from growing too large, which would push the softmax into regions with very small gradients.

### Multi-Head Attention

Rather than computing a single attention function, transformers use **multi-head attention**: the Q, K, and V projections are split into multiple parallel "heads," each learning to attend to different aspects of the input. One head might learn syntactic relationships (subject-verb agreement), another might learn semantic relationships (word meaning in context), and another might learn positional patterns. The outputs of all heads are concatenated and projected back to the model's dimension.

This mechanism allows the model to simultaneously capture multiple types of relationships between tokens, which is far more expressive than a single attention computation.

### Positional Encoding

Since the self-attention mechanism processes all tokens in parallel, it has no inherent notion of token order; the sentence "the cat sat on the mat" would produce the same attention scores as "mat the on sat cat the" without additional information. Positional encodings solve this by adding position information to each token's embedding before it enters the transformer layers.

The original transformer used fixed sinusoidal functions of different frequencies for positional encoding. Modern models like Llama use Rotary Position Embeddings (RoPE), which encode position information directly into the attention computation and generalize better to sequence lengths not seen during training.

### Layer Structure

Each transformer layer consists of:

1. **Multi-head self-attention**: tokens attend to each other
2. **Layer normalization**: stabilizes training by normalizing activations
3. **Feed-forward network**: a two-layer neural network applied independently to each token position, providing additional capacity for learning complex transformations
4. **Residual connections**: the input to each sub-layer is added to its output, which helps gradients flow during training and allows deeper networks

A typical LLM stacks dozens of these layers. For example, GPT-3 has 96 layers, and Llama 3 70B has 80 layers.

### Encoder-Only, Decoder-Only, and Encoder-Decoder Models

The original transformer used both an encoder and decoder for machine translation: the encoder processes the source language and the decoder generates the target language, attending to both its own output and the encoder's representation.

In practice, three variants have emerged:

- **Encoder-only** (e.g., BERT): Processes the full input bidirectionally. Excellent for understanding tasks like classification, named entity recognition, and semantic similarity. Each token can attend to all other tokens in both directions.
- **Decoder-only** (e.g., GPT, Claude, Gemini, Llama): Processes tokens left-to-right, where each token can only attend to tokens that came before it (causal masking). This is the architecture used by virtually all modern LLMs because it naturally supports text generation: predict the next token given all previous tokens.
- **Encoder-decoder** (e.g., T5, the original transformer): Uses both components. Still used for some translation and summarization tasks, but decoder-only models have largely superseded this approach for general-purpose language modeling.


## Tokenization

Before text can be processed by a transformer, it must be converted into a sequence of integers, a process called tokenization. The choice of tokenization strategy has significant practical implications for model performance, cost, and capability.

### Why Not Just Use Characters or Words?

Character-level tokenization produces very long sequences (a single word might be 5-10 tokens), which is computationally expensive because the self-attention mechanism scales quadratically with sequence length. Word-level tokenization has the opposite problem: the vocabulary must be enormous to cover all possible words, and any word not in the vocabulary (misspellings, technical jargon, foreign words) cannot be represented at all.

Modern tokenizers use a middle ground: **subword tokenization**, which splits text into chunks that balance vocabulary size against sequence length.

### Byte Pair Encoding (BPE)

BPE, used by GPT models and Llama, starts with individual characters and iteratively merges the most frequent adjacent pairs. For example, starting with the characters "l", "o", "w", the frequent pair "lo" would be merged into a single token, then "low" might be merged next. After many iterations, common words become single tokens while rare words are split into meaningful subword pieces.

The result is a vocabulary of typically 32,000 to 128,000 tokens. Common English words like "the" or "and" are single tokens. Less common words are split into recognizable pieces: "tokenization" might become "token" + "ization". This means the model can handle any text, including words it has never seen before, by decomposing them into known subword units.

### WordPiece and SentencePiece

**WordPiece**, used by BERT and related models, is similar to BPE but uses a slightly different merging criterion based on likelihood rather than frequency.

**SentencePiece** is a language-independent tokenizer that treats the input as a raw byte stream rather than pre-tokenized words. This makes it effective for multilingual models because it does not assume whitespace-separated words (important for languages like Chinese and Japanese). SentencePiece is used by models like T5 and Llama.

### Practical Implications of Tokenization

Understanding tokenization matters for practical use of LLMs:

- **Cost**: API providers charge per token, not per word. A word like "indistinguishable" might be 3-4 tokens, while "the" is one token. Roughly, one token is about 0.75 English words, or about 4 characters.
- **Context window**: Models have a maximum number of tokens they can process at once (the context window). GPT-4 supports up to 128K tokens; Gemini 2.5 supports up to 1M tokens. Understanding token counts helps you estimate how much text fits in the context window.
- **Non-English text**: Tokenizers trained primarily on English text tend to produce more tokens per word for other languages, which means non-English text uses more of the context window and costs more per word.
- **Code**: Programming languages tokenize differently from natural language. Variable names, operators, and whitespace all consume tokens.


## From Transformers to Large Language Models

The path from the transformer architecture to modern LLMs involved three key developments: scale, training methodology, and alignment.

### Scale

The most striking trend in LLM development has been the dramatic scaling of model size and training data. The original transformer (2017) had 65 million parameters. GPT-2 (2019) had 1.5 billion. GPT-3 (2020) jumped to 175 billion. Current frontier models are estimated to have hundreds of billions to over a trillion parameters, trained on trillions of tokens of text.

This scaling has revealed emergent capabilities: abilities that appear only once a model reaches a certain size. Small models can complete simple sentences; large models can write essays, debug code, solve math problems, and engage in multi-step reasoning. The precise mechanisms behind these emergent capabilities are an active area of research.

### Pre-training and the Next Token Prediction Objective

LLMs are pre-trained using a simple but powerful objective: predict the next token given all preceding tokens. The model reads vast amounts of text and learns to predict what comes next at each position. Despite the simplicity of this objective, it forces the model to learn grammar, facts, reasoning patterns, coding conventions, and much more, because predicting the next token accurately requires understanding the context deeply.

Pre-training is enormously expensive, typically requiring thousands of GPUs running for weeks or months, at a cost of millions to hundreds of millions of dollars. This is why only a handful of organizations train frontier models from scratch.

### Instruction Tuning and RLHF

A raw pre-trained model is a powerful text predictor but not a useful assistant. It will continue any text you give it, but it does not naturally follow instructions or engage in helpful dialogue. Two additional training stages transform it into a useful tool:

1. **Instruction tuning** (also called supervised fine-tuning or SFT): The model is trained on examples of instructions paired with high-quality responses. This teaches the model to follow directions rather than simply predict the next likely token.

2. **Reinforcement Learning from Human Feedback (RLHF)**: Human evaluators rank model outputs by quality, and these rankings are used to train a reward model. The LLM is then fine-tuned using reinforcement learning to produce outputs that score highly according to the reward model. This process improves helpfulness, reduces harmful outputs, and aligns the model's behavior with human preferences.

Some newer approaches, such as Direct Preference Optimization (DPO), achieve similar results without explicitly training a separate reward model.


## Key Capabilities of Modern LLMs

Modern LLMs exhibit several capabilities that emerge from their training:

### In-Context Learning

LLMs can learn new tasks from examples provided in the prompt, without any changes to the model's weights. By showing the model a few input-output examples (few-shot prompting), it can generalize and apply the pattern to new inputs. This is remarkable because the model is not being retrained; it is performing a form of learning purely through the attention mechanism at inference time.

### Chain-of-Thought Reasoning

When prompted to "think step by step," LLMs produce intermediate reasoning steps that lead to more accurate final answers, particularly for math, logic, and multi-step problems. This technique, called chain-of-thought prompting, significantly improves performance on tasks that require sequential reasoning.

Modern "reasoning models" like OpenAI's o-series and Google's Gemini 2.5 have been specifically trained to engage in extended internal reasoning before producing a response.

### Tool Use and Agents

LLMs can be trained to use external tools (search engines, calculators, code interpreters, APIs) by generating structured function calls. This allows them to overcome their inherent limitations (such as lack of real-time information or difficulty with precise arithmetic) by delegating to specialized tools.

AI agents take this further by using LLMs as the reasoning core of autonomous systems that can plan multi-step tasks, execute actions, observe results, and adapt their approach. This is an active and rapidly evolving area of development.

### Multimodal Understanding

Recent models accept not just text but also images, audio, and video as input. Google's Gemini and OpenAI's GPT-4o are natively multimodal, capable of analyzing photographs, reading charts, transcribing audio, and understanding video content. This expands LLMs from pure text processing to general-purpose perception and reasoning systems.


## Practical Considerations

When working with LLMs there are two fundamentally different approaches, each covered in its own chapter:

### Using Public APIs

Cloud providers offer access to frontier models through APIs. Google's Gemini, OpenAI's GPT models, and Anthropic's Claude are available through simple HTTP calls or Python client libraries. This approach requires no GPU hardware and gives you access to the most capable models available. The tradeoffs are cost (you pay per token), latency (network round-trips), privacy (your data is sent to a third party), and dependence on the provider's availability and pricing. We cover this approach in the next chapter, **LLMs with Public APIs**.

### Running Local Models

Open-weights models like Meta's Llama, Mistral, Qwen, and Google's Gemma can be downloaded and run on your own hardware. Tools like Ollama and llama.cpp make it straightforward to run quantized models on consumer hardware, including laptops with Apple Silicon. Local models offer privacy, no per-token cost, and the ability to fine-tune for specific tasks. The tradeoffs are that local models are generally less capable than frontier API models, and running larger models requires significant GPU memory. We cover this approach in the chapter **LLMs with Local Models**.

### Choosing Between APIs and Local Models

The choice depends on your requirements:

- **Privacy-sensitive data**: Use local models to keep data on your own hardware.
- **Maximum capability**: Use frontier API models (Gemini, Claude, GPT) for the most challenging tasks.
- **Cost at scale**: Local models have no per-token cost once the hardware is available, making them economical for high-volume applications.
- **Offline or edge deployment**: Local models can run without an internet connection.
- **Rapid prototyping**: APIs offer the fastest path from idea to working prototype.

Many production systems use a combination: a fast, inexpensive local model for routine tasks and an API call to a frontier model for complex ones.