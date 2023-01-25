# Recommendation Systems

Recommendation systems are a type of information filtering system that utilize historical data, such as past user behavior or interactions, to predict the likelihood of a user's interest in certain items or products. We will not write any recommendation systems from scratch in this chapter. We will review two open source recommendation systems that I have used at work. Both are based on TensorFlow.

Recommendation systems can use a wide variety of techniques, such as collaborative filtering, content-based filtering, and hybrid methods combining filtering algorithms, Rank Matrix or Deep Learning technologies, etc. to generate personalized recommendations for users. Collaborative filtering algorithms make recommendations based on the actions of similar users, while content-based filtering algorithms base recommendations on the attributes of items that a user has previously shown interest in. Hybrid methods may be further enhanced by incorporating additional data sources, such as demographic information, or by utilizing more advanced machine learning techniques, such as deep learning or reinforcement learning.

TBD

## TensorFlow Recommenders

We will refer to the documentation and examples [https://www.tensorflow.org/recommenders](https://www.tensorflow.org/recommenders) and follow the last Movie Lens example.

There are several types of data that could be used for recommending movies:

- User interactions (selecting movies).
- User data. User data is not used in this example, but for a product recommendation system I have created embeddings of all available data features associated with users.
- Movie data based on text embedding of movie titles. Note: for product recommendation systems you might still use text embedding of product descriptsions, but you would also likely create embeddings based on product features.




TBD

## A Transformer-based Recommendation System

Here we use [an example provided with Keras](https://keras.io/examples/structured_data/movielens_recommendations_transformers/) that was written by [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/) that transforms input training data to a textual representation for input to a Transformer model.

It may seem like we are asking a Transformer model to do a lot of work making product recommendations from textual data but given the great success of of the BERT, GPT-3, and ChatGPT Transformer models it is not surprising that Khalid Salama's model performs well. While I would use TensorFlow Recommenders in a new project, I find this Transformer-based recommendation system to be fascinating and it is something that I have revisted after using it for a prototype system at work.

### Overview of TBD

Khalid Salma based this example on the paper [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1905.06874) by Qiwei Chen, Huan Zhao, Wei Li, Pipei Huang, and Wenwu Ou.

TBD

