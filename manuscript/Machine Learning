# "Classic" Machine Learning (Optional Material)

Here we cover just a single example of what I think of as "classic machine learning" using the [scikit-learn](https://scikit-learn.org/stable/) Python library. Later we cover deep learning in three separate chapters. Deep learning models are more general and powerful but it is important to recognize the types of problems that can be solved using the simpler techniques.

They only requirements for this chapter is **pip install scikit-learn pandas**.

Please note that the content in this book is heavily influenced by what I use in my own work. I mostly use deep learning so its coverage comprises half this book. For this classic machine learning chapter I only use a classification model. I will not be covering regression or clustering models.

We will use the same Wisconsin cancer dataset for both the following classification example and a deep learning classification example in a later chapter. Here are the first few rows of the file **labeled_cancer_data.csv**:

![](wisconsindata-1.png)

The last column **class* indicates the class of the sample, 0 for non-malignant and 1 for malignant. The scikit-learn library has high level and simple to use utilities for reading CSV (spreadsheet) data and for preparing the data for training and testing. I don't use these utilities here because I am reusing the data loading code from the later deep learning example.

Listing of **load_data.py**:

```python
import pandas

def load_data():

    train_df = pandas.read_csv("labeled_cancer_data.csv")
    test_df = pandas.read_csv("labeled_test_data.csv")

    train = train_df.values
    X_train = train[:,0:9].astype(float) # 9 inputs
    print("Number training examples:", len(X_train))
    # Training data: one target output (0 for no cancer, 1 for malignant)
    Y_train = train[:,-1].astype(float)
    test = test_df.values
    X_test = test[:,0:9].astype(float)
    Y_test = test[:,-1].astype(float)
    print("Number testing examples:", len(X_test))
    return (X_train, Y_train, X_test, Y_test)
```

TBD

## Classification Models

TBD

```python
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from load_data import load_data

(X_train, Y_train, X_test, Y_test) = load_data()

# Remove mean and scale to unit variance:
scaler1 = StandardScaler()
scaler2 = StandardScaler()

X_train = scaler1.fit_transform(X_train)
X_test = scaler2.fit_transform(X_test)

# Use the KNN classifier to fit data:
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)

# Predict y data with classifier: 
y_predict = classifier.predict(X_test)

# Print results: 
print(confusion_matrix(Y_test, y_predict))
print(classification_report(Y_test, y_predict))
```

TBD

```bash
$ python classification.py 
Number training examples: 677
Number testing examples: 15
[[7 0]
 [1 7]]
              precision    recall  f1-score   support

         0.0       0.88      1.00      0.93         7
         1.0       1.00      0.88      0.93         8

    accuracy                           0.93        15
   macro avg       0.94      0.94      0.93        15
weighted avg       0.94      0.93      0.93        15
```

## Classic Machine Learning Wrapup

I have already admitted my personal biases in favor of deep learning over simpler machine learning and I proved that by using perhaps only 1% of the functionality of scikit-learn in this chapter.

