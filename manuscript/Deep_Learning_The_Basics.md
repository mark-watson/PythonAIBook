# Part III - Deep Learning

In the next two chapters we explore some basic theory underlying deep learning and then look at practical examples building models from spreadsheet data, performing natural language processing (NLP) tasks, and have fun with models to generate images from text.

When you have finished reading this section and want to learn more about specific deep learning architectures I recommend using this up to date list of short descriptive papers [https://github.com/dair-ai/ML-Papers-Explained](https://github.com/dair-ai/ML-Papers-Explained).


# The Basics of Deep Learning

Deep learning is a subfield of machine learning that is concerned with the design and implementation of artificial neural networks (ANNs) with multiple layers, also known as deep neural networks (DNNs). These networks are inspired by the structure and function of the human brain, and are designed to learn from large amounts of data such as images, text, and audio.

A neural network consists of layers of interconnected nodes, or neurons, which are organized into an input layer, one or more hidden layers, and an output layer. Each neuron receives input from the neurons in the previous layer, performs a computation, and passes the result to the neurons in the next layer. The computation typically involves a dot product of the input with a set of weights and an activation function, which is a non-linear function applied to the result. The weights are the parameters of the network that are learned during training.

The basic building block of a deep neural network is an artificial neuron, also known as perceptron, which is a simple mathematical model for a biological neuron. A perceptron receives input from other neurons and it applies a linear transformation to the input, followed by a non-linear activation function.

Deep learning networks can be feedforward networks where the data flows in one direction from input to output, or recurrent networks where the data can flow in a cyclic fashion.

There are different types of deep learning architectures such as feedforward neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs) and Generative Adversarial Networks (GANs). These architectures are designed to learn specific types of features and patterns from different types of data.

Deep learning models are trained using large amounts of labeled data, and typically use supervised or semi-supervised learning techniques. The training process involves adjusting the weights of the network to minimize a loss function, which measures the difference between the predicted output and the true output. This process is known as back-propagation, which is an algorithm for training the weights in a neural network by propagating the error back through the network. In the first AI book I wrote in the 1980s I covered the implementation of back-propagation in detail. As I write the material here on deep learning I think that it is more important for you to have the skills to choose appropriate tools for different applications and be less concerned about low-level implementation details. I think this characterizes the change in trajectory of AI from being about tool building to the skills of using available tools and sometimes previously trained models while spending more of your effort analyzing business functions and in general application domains.

Deep Learning has been applied to various fields such as Computer Vision, Natural Language Processing, Speech Recognition, etc.

## Using TensorFlow and Keras for Building a Cancer Prediction Model

Please [follow this link to Google Colab](https://colab.research.google.com/drive/18UJ-5i6_SyfU01PptfNxvR3ZCNirrRgc?usp=sharing) to see the example using TensorFlow to build a model of the University of Wisconsin cancer dataset. A subset of this Jupyter notebook can also be found in the file **deep-learning/wisconsin_data_github.py** but you will need to install all dependencies automatically installed by Colab and you might need to remove the calls to TensorBoard.

We use the Python package [skimpy](https://pypi.org/project/skimpy/) that is a lightweight tool for creating summary statistics from [Pandas](https://pandas.pydata.org) dataframes. Please note that I will use Pandas dataframes without much explaination so if you have never used Pandas please review [the tutorial on importing and using CSV spreadsheet data](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html#min-tut-02-read-write). The other tutorials are optional.

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

You can compare this example using TensorFlow and Keras to our similar classification example using the same data where we used  the **Scikit-learn** library.


### PyTorch and JAX

In addition to TensorFlow/Keras, the two other most popular frameworks for deep learning are [PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html) and [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).

All three frameworks are popular and well supported. I started studying deep learning at the same time that TensorFlow was initially released and I use TensorFlow (usually with the easier to use Keras APIs) for at least 90% of my professional deep learning work. Because of my own history I am showing you TensorFlow/Keras examples but if PyTorch or JAX appeal more to you then by all means use the framework that fits your requirements.

