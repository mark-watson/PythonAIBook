# Overview of Recommendation Systems

Recommendation systems are a type of information filtering system that utilize historical data, such as past user behavior or interactions, to predict the likelihood of a user's interest in certain items or products. As an example application: if a product web site has 100K products that is too many for customers to browse through. Based on a customers past purchases, finding customers with similar purchases, etc. it is possible to filter the products shown to a customer.

Writing recommendation systems is a common requirement for almost all businesses that sell products to customers. Before we get started we need to define two terms that you may not be familiar with: [Collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering): uses both similarities between users and items to calculate recommendations. This linked Wikipedia article also discusses content-based filtering which uses user and item features.

The [Movie Lens dataset](https://grouplens.org/datasets/movielens/) created by the GroupLens Research organization uses the user movie preference [https://movielens.org](https://movielens.org) dataset. This dataset is a standard for developing and evaluating recommendation system algorithms and models.

There are at least three good approaches to take:

- Use a turnkey recommendation system like [Amazon Personalize](https://aws.amazon.com/personalize/) that is a turn-key service on AWS. You can evaluate Amazon Personalize for your company's use by spending about one hour working through the [getting started tutorial](https://github.com/aws-samples/amazon-personalize-samples).
- Use one of the standard libraries or TensorFlow implementations for the classic approach using [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems) for collaborative filtering. Examples are Eric Lundquist's Python library [rankfm](https://github.com/etlundquist/rankfm) and the first example for the [TensorFlow Recommenders library](https://www.tensorflow.org/recommenders/examples/quickstart). Google has a [good Matrix Factorization tutorial](https://developers.google.com/machine-learning/recommendation/collaborative/matrix). While I prefer using the much more complicated TensorFlow Recommenders library, using matrix factorization is probably a good way to start and I recommend taking an hour to work through [this Google Colab tutorial](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/recommendation-systems/recommendation-systems.ipynb)
- Use the [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) library that supports multi-tower deep learning models that use data for user interactions, user detail data, and product detail data.

We will not write any recommendation systems from scratch in this chapter. We will review one open source recommendation system that I have used at work, the TensorFlow Recommenders library.

Recommendation systems can use a wide variety of techniques, such as collaborative filtering, content-based filtering, and hybrid methods combining filtering algorithms, Matrix Factorization, or Deep Learning technologies, etc. to generate personalized recommendations for users. Collaborative filtering algorithms make recommendations based on the actions of similar users, while content-based filtering algorithms base recommendations on the attributes of items that a user has previously shown interest in. Hybrid methods may be further enhanced by incorporating additional data sources, such as demographic information, or by utilizing more advanced machine learning techniques, such as deep learning or reinforcement learning.

## TensorFlow Recommenders

I used Google's TensorFlow Recommenders library for a work project. I recommend it because it has very good documentation, many examples using the Movie Lens dataset, and is fairly easy to adapt to general user/product recommendation systems.

We will refer to the documentation and examples [https://www.tensorflow.org/recommenders](https://www.tensorflow.org/recommenders) and follow the last Movie Lens example.

There are several types of data that could be used for recommending movies:

- User interactions (selecting movies).
- User data. User data is not used in this example, but for a product recommendation system I have created embeddings of all available data features associated with users.
- Movie data based on text embedding of movie titles. Note: for product recommendation systems you might still use text embedding of product descriptions, but you would also likely create embeddings based on product features.

We will use the [TensorFlow Recommenders using rich features example](https://www.tensorflow.org/recommenders/examples/deep_recommenders). For the following overview discussion, you may want to either open this link to read this example or open the alternative [Google Colab example link](https://colab.research.google.com/github/tensorflow/recommenders/blob/main/docs/examples/deep_recommenders.ipynb) to run the example on Colab. Please note that this example takes about ten minutes to run on Colab.  For our discussion, I will use short code snippets and use one screenshot of the example in Colab so you can optionally just follow along without opening either link for now.

The TF recommenders example starts with reading the [Movie Lens](https://grouplens.org/datasets/movielens/) dataset using the TensorFlow Data library:

```python
import tensorflow_datasets as tfds
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])
```

We need to later generate embedding layers for both unique movie  titles and also unique user IDs. We start with getting sequences for unique movie titles and user IDs:

```python
unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))
```
    
In this example Python user and movie models are derived from the Python class **tf.keras.Model**. Let's look at the implementation of these two models:

```python
class UserModel(tf.keras.Model):
  
  def __init__(self):
    super().__init__()

    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
    ])
    self.timestamp_embedding = tf.keras.Sequential([
        tf.keras.layers.Discretization(timestamp_buckets.tolist()),
        tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
    ])
    self.normalized_timestamp = tf.keras.layers.Normalization(
        axis=None
    )

    self.normalized_timestamp.adapt(timestamps)

  def call(self, inputs):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
    ], axis=1)
```

The function **tf.keras.layers.StringLookup** is used to create an embedding layer from a sequence of unique string IDs. Timestamps for user selection events are fairly continuous so we use **tf.keras.layers.Discretization** to collapse a wide range of timestamp values into discrete bins.

Classed derived from class **tf.keras.Model** are expected to implement a **call** method that is passed a inputs and returns a single Tensor of concatenated inputs and timestamp embeddings.

We build a similar model for movies:

```python
class MovieModel(tf.keras.Model):
  
  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.title_embedding = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
          vocabulary=unique_movie_titles,mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
    ])

    self.title_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens)

    self.title_text_embedding = tf.keras.Sequential([
      self.title_vectorizer,
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

    self.title_vectorizer.adapt(movies)

  def call(self, titles):
    return tf.concat([
        self.title_embedding(titles),
        self.title_text_embedding(titles),
    ], axis=1)
```

The class **MovieModel** is different than the class **UserModel** since we create embeddings for movie titles instead of IDs.

We also wrap the user model in a separate query model that combines a user model with dense fully connected layers:

```python
class QueryModel(tf.keras.Model):
  """Model for encoding user queries."""

  def __init__(self, layer_sizes):
    """Model for encoding user queries.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    # We first use the user model for generating embeddings.
    self.embedding_model = UserModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))
    
  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)
```

The **call** method returns the values calculated from feeding the input layer into the dense fully connected layers that have a *relu* non-linear activation function.

We also wrap the movie model in a candidate recommendation model:

```python
class CandidateModel(tf.keras.Model):
  """Model for encoding movies."""

  def __init__(self, layer_sizes):
    """Model for encoding movies.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    self.embedding_model = MovieModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))
    
  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)
```

The **call** method returns the values calculated from feeding the input layer for a movie model into the dense fully connected layers that have a *relu* non-linear activation function.

We finally train a deep learning model by creating an instance of class **MovielensModel** and calling its inherited **fit** method:

```python
model = MovielensModel([64, 32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

two_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=num_epochs,
    verbose=0)

accuracy = two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")
```

The output looks like:

```
Top-100 accuracy: 0.29.
```

This top-100 accuracy means that if you make a movie recommendation, it has a 29% chance of being in the top 100 recommendations for a user.
 
We can plot the training accuracy vs. training epoch for both one and two layers:

![](tfr.png)

The example Google Colab project has an additional training run that gets better accuracy by stacking many additional hidden layers in the user and movie wrapper Python classes.


## Recommendation Systems Wrap-up

If you need to write a recommendation system for your work then I hope this short overview chapter will get you started. Here are alternative approaches and a few resources:

- Consider using [Amazon Personalize](https://aws.amazon.com/personalize/) which is a turn-key service on AWS.
- Consider using Google's turn-key [Recommendations AI](https://cloud.google.com/recommendations).
- Eric Lundquist has written a Python library [rankfm](https://github.com/etlundquist/rankfm) for factorization machines for recommendation and ranking problems.
- If product data includes pictures then consider using this [Keras example](https://keras.io/examples/nlp/nl_image_search/) as a guide for creating embeddings for images and implementing image search.
- A research idea: [a Keras example](https://keras.io/examples/structured_data/movielens_recommendations_transformers/) that was written by Khalid Salama that transforms input training data to a textual representation for input to a Transformer model. This example is based on the paper [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874) by Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou.


# Book Wrap-up

Thank you, dear reader, for spending the time for taking an adventure with me: exploring AI programming ideas using Python. Because some of the material will be dated quickly, I was motivated to write this book and release it quickly. Indeed, I wrote this book in just a two and a half month period.

My first ten books were published as conventional print books by McGraw-Hill, Springer-Verlag, John Riley, and other publishers. I then published several books with free licenses that can be downloaded from [https://markwatson.com](https://markwatson.com). My seven eBooks subsequently [published on Leanpub](https://leanpub.com/u/markwatson) have cumulatively been updated many times to new editions. I would like to that the Leanpub platform for simplifying the process of writing and frequently updating eBooks.

I tend to remove some material in new eBook editions that might be outdated, and add new chapters on different topics. Because of this, when you get new editions of my books (free on Leanpub), consider also keeping the old editions.

I live in Sedona Arizona. If you are every passing through Sedona, contact me (see [https://markwatson.com](https://markwatson.com)) if you would like to have coffee and talk about Python and AI.