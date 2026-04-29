import pandas as pd


def load_data():
    train_df = pd.read_csv("labeled_cancer_data.csv")
    test_df = pd.read_csv("labeled_test_data.csv")

    train = train_df.to_numpy()
    X_train = train[:, 0:9].astype(float)  # 9 input features
    print("Number training examples:", len(X_train))
    # Training target: one output (0 for non-malignant, 1 for malignant)
    Y_train = train[:, -1].astype(float)

    test = test_df.to_numpy()
    X_test = test[:, 0:9].astype(float)
    Y_test = test[:, -1].astype(float)
    print("Number testing examples:", len(X_test))
    return (X_train, Y_train, X_test, Y_test)
