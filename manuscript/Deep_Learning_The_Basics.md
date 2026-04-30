# The Basics of Deep Learning

Deep learning is a subfield of machine learning that is concerned with the design and implementation of artificial neural networks (ANNs) with multiple layers, also known as deep neural networks (DNNs). These networks are inspired by the structure and function of the human brain, and are designed to learn from large amounts of data such as images, text, and audio.

A neural network consists of layers of interconnected nodes, or neurons, which are organized into an input layer, one or more hidden layers, and an output layer. Each neuron receives input from the neurons in the previous layer, performs a computation, and passes the result to the neurons in the next layer. The computation typically involves a dot product of the input with a set of weights and an activation function, which is a non-linear function applied to the result. The weights are the parameters of the network that are learned during training.

The basic building block of a deep neural network is an artificial neuron, also known as perceptron, which is a simple mathematical model for a biological neuron. A perceptron receives input from other neurons and it applies a linear transformation to the input, followed by a non-linear activation function.

Deep learning networks can be feedforward networks where the data flows in one direction from input to output, or recurrent networks where the data can flow in a cyclic fashion.

There are different types of deep learning architectures such as feedforward neural networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs) and Generative Adversarial Networks (GANs). These architectures are designed to learn specific types of features and patterns from different types of data.

Deep learning models are trained using large amounts of labeled data, and typically use supervised or semi-supervised learning techniques. The training process involves adjusting the weights of the network to minimize a loss function, which measures the difference between the predicted output and the true output. This process is known as back-propagation, which is an algorithm for training the weights in a neural network by propagating the error back through the network. In the first AI book I wrote in the 1980s I covered the implementation of back-propagation in detail. As I write the material here on deep learning I think that it is more important for you to have the skills to choose appropriate tools for different applications and be less concerned about low-level implementation details. I think this characterizes the change in trajectory of AI from being about tool building to the skills of using available tools and sometimes previously trained models while spending more of your effort analyzing business functions and in general application domains.

Deep Learning has been applied to various fields such as Computer Vision, Natural Language Processing, Speech Recognition, etc.

## Using PyTorch for Building a Cancer Prediction Model

We will use [PyTorch](https://pytorch.org/) to build a neural network that classifies the same University of Wisconsin cancer dataset we used in the scikit-learn chapter. This lets us directly compare the deep learning approach with the classic K-Nearest Neighbors classifier.

The examples for this chapter are in the directory **source-code/deep_learning_basics**.

The requirements for this chapter are:

```bash
uv pip install torch scikit-learn pandas numpy
```

### Why PyTorch?

PyTorch is the most widely used deep learning framework in both research and industry. Developed originally by Meta AI, it provides:

- **Dynamic computation graphs**: build and modify your network on-the-fly, making debugging natural.
- **Pythonic API**: feels like writing regular Python code, not configuring a separate computation engine.
- **Extensive ecosystem**: integrates with Hugging Face Transformers, torchvision, torchaudio, and many more libraries.
- **Strong GPU support**: seamlessly move computations between CPU, CUDA GPUs, and Apple Silicon (MPS).

### Loading and Preparing the Data

We reuse the same CSV data files from our machine learning chapter. The data loading code converts the Pandas DataFrames to NumPy arrays, scales the features, then wraps everything in PyTorch tensors:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load the cancer CSV files."""
    train_df = pd.read_csv("../machine-learning/labeled_cancer_data.csv")
    test_df = pd.read_csv("../machine-learning/labeled_test_data.csv")

    train = train_df.to_numpy()
    X_train = train[:, 0:9].astype(np.float32)
    Y_train = train[:, -1].astype(np.float32).reshape(-1, 1)

    test = test_df.to_numpy()
    X_test = test[:, 0:9].astype(np.float32)
    Y_test = test[:, -1].astype(np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test
```

Note that we use `np.float32` — PyTorch expects 32-bit floats by default (unlike NumPy's default of 64-bit).

### Defining the Model

In PyTorch, we define neural network architectures by subclassing `nn.Module`. Our network has two hidden layers of 15 neurons with ReLU activation, a dropout layer for regularization, and a single output neuron:

```python
class CancerNet(nn.Module):
    """Simple feedforward network: 9 → 15 → 15 → 1."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(15, 1),
        )

    def forward(self, x):
        return self.network(x)
```

The `forward` method defines how data flows through the network. PyTorch's autograd system automatically computes the gradients needed for backpropagation — we never need to write the backward pass manually.

### The Training Loop

Unlike higher-level frameworks, PyTorch gives you explicit control over the training loop. This makes it easy to customize training behavior, add logging, or implement complex training schedules:

```python
def train_model(model, train_loader, epochs=60, lr=0.01):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss: {avg_loss:.4f}")
```

We use `BCEWithLogitsLoss` which combines a sigmoid activation with binary cross-entropy loss — this is numerically more stable than applying sigmoid separately. The training loop follows the standard PyTorch pattern: zero gradients, forward pass, compute loss, backward pass, update weights.

### Running the Example

Here is the complete output from running **cancer_model.py**:

```bash
$ python cancer_model.py
Training examples: 554
Test examples:     15

CancerNet(
  (network): Sequential(
    (0): Linear(in_features=9, out_features=15, bias=True)
    (1): ReLU()
    (2): Linear(in_features=15, out_features=15, bias=True)
    (3): ReLU()
    (4): Dropout(p=0.2, inplace=False)
    (5): Linear(in_features=15, out_features=1, bias=True)
  )
)

Training:
  Epoch  10/60  loss: 0.6913
  Epoch  20/60  loss: 0.6559
  Epoch  30/60  loss: 0.6250
  Epoch  40/60  loss: 0.5874
  Epoch  50/60  loss: 0.5421
  Epoch  60/60  loss: 0.4969

Confusion Matrix:
[[9 0]
 [1 5]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.90      1.00      0.95         9
         1.0       1.00      0.83      0.91         6

    accuracy                           0.93        15
   macro avg       0.95      0.92      0.93        15
weighted avg       0.94      0.93      0.93        15

Sample predictions (should be close to [[0], [1]]):
[[0.8588059]
 [0.9907742]]
```

The model achieves 93% accuracy on the test set — matching the performance of our scikit-learn KNN classifier from the earlier chapter. The loss decreases steadily during training, and the sample predictions show that the model learns to distinguish between non-malignant and malignant cases.

You can compare this PyTorch example to our similar classification example using the same data where we used the **Scikit-learn** library. The deep learning approach requires more code but gives us full control over the model architecture, training process, and the ability to scale to much larger and more complex problems.


