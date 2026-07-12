"""
Cancer prediction model using PyTorch.

Builds a feedforward neural network to classify the
Wisconsin Breast Cancer dataset (same data used in the
scikit-learn chapter).
"""

from typing import override

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# ── Data loading ──────────────────────────────────────


def load_data():
    """Load the cancer CSV files from the
    machine-learning directory."""
    train_df = pd.read_csv("../machine-learning/labeled_cancer_data.csv")
    test_df = pd.read_csv("../machine-learning/labeled_test_data.csv")

    train = train_df.to_numpy()
    # First 9 columns are features
    X_train = train[:, 0:9].astype(np.float32)
    # Last column is the label (0 or 1)
    Y_train = train[:, -1].astype(np.float32).reshape(-1, 1)

    test = test_df.to_numpy()
    # First 9 columns are features
    X_test = test[:, 0:9].astype(np.float32)
    # Last column is the label (0 or 1)
    Y_test = test[:, -1].astype(np.float32).reshape(-1, 1)

    # Scale features to zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, Y_train, X_test, Y_test


# ── Model definition ─────────────────────────────────


class CancerNet(nn.Module):
    """Simple feedforward network: 9 → 15 → 15 → 1."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Input layer: 9 features → 15 hidden units
            nn.Linear(9, 15),
            nn.ReLU(),
            # Hidden layer: 15 → 15
            nn.Linear(15, 15),
            nn.ReLU(),
            # Dropout for regularization (20%)
            nn.Dropout(0.2),
            # Output layer: 1 logit (binary classification)
            nn.Linear(15, 1),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Training loop ────────────────────────────────────


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    epochs: int = 60,
    lr: float = 0.01,
) -> None:
    """Train the model using SGD and BCEWithLogitsLoss."""
    # Binary cross-entropy with built-in sigmoid
    criterion = nn.BCEWithLogitsLoss()
    # Stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Reset gradients from previous step
            optimizer.zero_grad()
            # Forward pass: compute predictions
            outputs = model(X_batch)
            # Compute loss between predictions and labels
            loss = criterion(outputs, y_batch)
            # Backward pass: compute gradients
            loss.backward()
            # Update model weights
            optimizer.step()
            total_loss += loss.item()

        # Print average loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss: {avg_loss:.4f}")


# ── Main ─────────────────────────────────────────────

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data()
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples:     {len(X_test)}\n")

    # Convert numpy arrays to PyTorch tensors
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    train_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        train_ds, batch_size=100, shuffle=True
    )

    model = CancerNet()
    print(model)
    print()

    print("Training:")
    train_model(model, train_loader, epochs=60)

    # ── Evaluate on the test set ──
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test)
        # Get raw logit outputs from the model
        logits = model(X_test_t)
        # Convert logits to probabilities via sigmoid
        probs = torch.sigmoid(logits)
        # Threshold at 0.5 to get binary predictions
        predictions = (probs >= 0.5).float().numpy()

    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(Y_test, predictions))

    # ── Predict two individual samples ──
    # Sample 1: low values → expect benign (0)
    # Sample 2: high values → expect malignant (1)
    samples = torch.tensor(
        [
            [4, 1, 1, 3, 2, 1, 3, 1, 1],
            [3, 7, 7, 4, 4, 9, 4, 8, 1],
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        # Convert logits to probabilities for samples
        sample_probs = torch.sigmoid(model(samples))
    print("Sample predictions (should be close to [[0], [1]]):")
    print(sample_probs.numpy())
