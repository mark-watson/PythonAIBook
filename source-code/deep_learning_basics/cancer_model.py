"""
Cancer prediction model using PyTorch.

Builds a feedforward neural network to classify the Wisconsin
Breast Cancer dataset (same data used in the scikit-learn chapter).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# ── Data loading ──────────────────────────────────────────────

def load_data():
    """Load the cancer CSV files from the machine-learning directory."""
    train_df = pd.read_csv("../machine-learning/labeled_cancer_data.csv")
    test_df = pd.read_csv("../machine-learning/labeled_test_data.csv")

    train = train_df.to_numpy()
    X_train = train[:, 0:9].astype(np.float32)
    Y_train = train[:, -1].astype(np.float32).reshape(-1, 1)

    test = test_df.to_numpy()
    X_test = test[:, 0:9].astype(np.float32)
    Y_test = test[:, -1].astype(np.float32).reshape(-1, 1)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test


# ── Model definition ─────────────────────────────────────────

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


# ── Training loop ─────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples:     {len(X_test)}\n")

    # Convert to PyTorch tensors
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(Y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)

    model = CancerNet()
    print(model)
    print()

    print("Training:")
    train_model(model, train_loader, epochs=60)

    # ── Evaluate ──
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test)
        logits = model(X_test_t)
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float().numpy()

    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(Y_test, predictions))

    # ── Predict two individual samples ──
    samples = torch.tensor([
        [4, 1, 1, 3, 2, 1, 3, 1, 1],
        [3, 7, 7, 4, 4, 9, 4, 8, 1],
    ], dtype=torch.float32)
    with torch.no_grad():
        sample_probs = torch.sigmoid(model(samples))
    print("Sample predictions (should be close to [[0], [1]]):")
    print(sample_probs.numpy())
