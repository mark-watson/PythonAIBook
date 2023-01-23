# Part III - Deep Learning

In the next two chapters we explore some basic theory underlying deep learning and then look at practical examples building models from spreadsheet data, performing natural language processing (NLP) tasks, and have fun with models to generate images from text.

When you are done reading this section, and want to learn more about specific deep learning architectures I recommend using this up to date list of short descriptive papers [https://github.com/dair-ai/ML-Papers-Explained](https://github.com/dair-ai/ML-Papers-Explained).


# The Basics of Deep Learning

TBD

## Using TensorFlow and Keras for Building a Cancer Prediction Model

Please [follow this link to Google Colab](https://colab.research.google.com/drive/18UJ-5i6_SyfU01PptfNxvR3ZCNirrRgc?usp=sharing) to see the example using TensorFlow to build a model of the University of Wisconsin cancer dataset. A subset of this Jupyter notebook can also be found in the file **deep-learning/wisconsin_data_github.py** but you will need to install all dependencies and the calls to TensorBoard are removed.

We use the Python package [skimpy](https://pypi.org/project/skimpy/) that is a light weight tool for creating summary statistics from [Pandas](https://pandas.pydata.org) dataframes. Please note that I will use Pandas dataframes without much explaination so if you have never used Pandas please review [the tutorial on importing and using CSV spreadsheet data](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html#min-tut-02-read-write). The other tutorials are optional.

We use the **skimpy** library to get information on our dataset:

```python
train_uri = "https://raw.githubusercontent.com/mark-watson/cancer-deep-learning-model/master/train.csv"
test_uri = "https://raw.githubusercontent.com/mark-watson/cancer-deep-learning-model/master/test.csv"

train_df = pandas.read_csv(train_uri, header=None)

skim(train_df)
```

![Skimpy librarytool for creating summary statistics](skimpy.png)

We need to prepare the training and test data:

```python
train = train_df.values
X_train = train[:,0:9].astype(float) # 9 inputs
print("Number training examples:", len(X_train))
Y_train = train[:,-1].astype(float)  # one target output (0 for no cancer, 1 for malignant)
test = pandas.read_csv(test_uri, header=None).values
X_test = test[:,0:9].astype(float)
Y_test = test[:,-1].astype(float)
```

We now need to define the TensorFlow/Keras model architecture and train the model:

```python
model = Sequential()
model.add(Dense(tf.constant(15), input_dim=tf.constant(9), activation='relu'))
model.add(Dense(tf.constant(15), input_dim=tf.constant(15), activation='relu'))
model.add(Dropout(0.2)),
model.add(Dense(tf.constant(1), activation='sigmoid'))
model.summary()

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['accuracy'])

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [TensorBoard(log_dir=logdir,histogram_freq=1,write_graph=True, write_images=True)]

model.fit(X_train, Y_train, batch_size=100, epochs=60, callbacks=callbacks)
```

We use the trained model to make predictions on test data:

```python
y_predict = model.predict([[4,1,1,3,2,1,3,1,1], [3,7,7,4,4,9,4,8,1]])
print("* y_predict (should be close to [[0], [1]]):", y_predict)

* y_predict (should be close to [[0], [1]]): [[0.37185097]
 [0.9584093 ]]
```

TBD

### PyTorch and JAX

In addition to TensorFlow/Keras, the two other most popular frameworks for deep learning are [PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html) and [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).

All three frameworks are popular and well supported. I started studying deep learning at the same time that TensorFlow was initially released and I use TensorFlow (usually with the easier to use Keras APIs) for at least 90% of my professional deep learning work. Because of my own history I am showing you TensorFlow/Keras examples but if PyTorch or JAX appeal more to you then by all means use the framework that fits your requirements.


## Recommendation Systems

Writing recommendation systems is a common requirment for almost all businesses that sell products to customers. Before we get started we need to define a few terms that you may not be familiar with:

- Movie Lens dataset: TBD
- Collaborative filtering: TBD
- TBD

There are at least three good approaches to take:

- Use a turnkey recommendation system like [Amazon Personalize](https://aws.amazon.com/personalize/) that is a turn-key service on AWS. You can evaluate Amazon Personalize for your company's use by spending about one hour working through the [getting started tutorial](https://github.com/aws-samples/amazon-personalize-samples).
- Use one of the standard libraries or TensorFlow implementations for the classic approach using [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems) for collaborative filtering. Examples are Eric Lundquist's Python library [rankfm](https://github.com/etlundquist/rankfm) and the first example for the [TensorFlow Recommenders library](https://www.tensorflow.org/recommenders/examples/quickstart). Google has a [good Matrix Factorization tutorial](https://developers.google.com/machine-learning/recommendation/collaborative/matrix). While I prefer using the much more complicated TensorFlow Recommenders library, using matrix factorization is probably a good way to start and I recommend taking an hour to work through [this Google Colab tutorial](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/recommendation-systems/recommendation-systems.ipynb)
- Use the [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) library that supports multi-tower deep learning models that use data for user interactions, user detail data, and product detail data.

TBD
