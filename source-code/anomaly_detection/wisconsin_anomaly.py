"""
Wisconsin Breast Cancer anomaly-detection example.

Loads the cleaned Wisconsin Diagnostic Breast Cancer
dataset, preprocesses it (log-transform + min–max scaling,
matching the Java version), splits into training /
cross-validation / test sets, and evaluates two detectors:

  1. GaussianAnomalyDetector  (from-scratch statistics)
  2. IsolationForestDetector  (scikit-learn)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

from anomaly_detection import (
    GaussianAnomalyDetector,
    IsolationForestDetector,
    evaluate,
)

DATA_PATH = "cleaned_wisconsin_cancer_data.csv"

FEATURE_NAMES = [
    "Clump Thickness",
    "Uniformity of Cell Size",
    "Uniformity of Cell Shape",
    "Marginal Adhesion",
    "Single Epithelial Cell Size",
    "Bare Nuclei",
    "Bland Chromatin",
    "Normal Nucleoli",
    "Mitoses",
]


# ── data loading & preprocessing ─────────────────────────────


def load_wisconsin_data():
    """Load CSV, apply the same log-transform + min–max
    normalisation used in the Java version, and split
    into train / cross-validation / test."""

    raw = np.genfromtxt(DATA_PATH, delimiter=",")
    X_raw = raw[:, :9] * 0.1           # scale to [0, 1]

    # log-transform to approximate Gaussian shape
    X_log = np.log(X_raw + 1.2)
    row_min = X_log.min(axis=1, keepdims=True)
    row_max = X_log.max(axis=1, keepdims=True)
    X = (X_log - row_min) / (row_max - row_min + 1e-10)

    # Target: original column is 2 (benign) or 4 (malignant)
    # Map to 0 = normal, 1 = anomaly (malignant)
    y = ((raw[:, 9] - 2) * 0.5).astype(int)

    # Split: ~60% train, ~20% CV, ~20% test
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_train = int(0.6 * len(X))
    n_cv = int(0.2 * len(X))

    train_idx = idx[:n_train]
    cv_idx = idx[n_train:n_train + n_cv]
    test_idx = idx[n_train + n_cv:]

    # Training set: keep mostly normal (benign) examples,
    # allow ~10% anomalies through (matches Java logic)
    normal_mask = y[train_idx] == 0
    anomaly_mask = y[train_idx] == 1
    keep_anomaly = rng.random(anomaly_mask.sum()) < 0.1
    keep_idx = np.concatenate([
        train_idx[normal_mask],
        train_idx[anomaly_mask][keep_anomaly],
    ])
    X_train = X[keep_idx]

    return (
        X_train,                    # train features (mostly normal)
        X[cv_idx], y[cv_idx],       # cross-validation
        X[test_idx], y[test_idx],   # test
        X, y,                       # full set (for histograms)
    )


# ── histogram visualisation ──────────────────────────────────


def plot_feature_histograms(X, y, path="histograms.png"):
    """Save a 3×3 grid of feature histograms, colour-coded
    by class (normal / anomaly)."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle(
        "Wisconsin Cancer Dataset — Feature Distributions",
        fontsize=14, fontweight="bold",
    )

    for i, ax in enumerate(axes.flat):
        ax.hist(
            X[y == 0, i], bins=20, alpha=0.6,
            label="Normal", color="#4C72B0",
        )
        ax.hist(
            X[y == 1, i], bins=20, alpha=0.6,
            label="Anomaly", color="#DD4E4E",
        )
        ax.set_title(FEATURE_NAMES[i], fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150)
    print(f"\nHistograms saved to {path}")


# ── main ─────────────────────────────────────────────────────


def main():
    (X_train, X_cv, y_cv,
     X_test, y_test, X_all, y_all) = load_wisconsin_data()

    print(f"Training examples  : {len(X_train)}")
    print(f"Cross-val examples : {len(X_cv)}")
    print(f"Test examples      : {len(X_test)}")

    # Histograms
    plot_feature_histograms(X_all, y_all)

    # ── Approach 1: Gaussian statistical model ───────────
    gauss = GaussianAnomalyDetector()
    gauss.fit(X_train, y_cv, X_cv)
    y_pred_gauss = gauss.predict(X_test)
    evaluate("Gaussian Statistical Detector", y_test, y_pred_gauss)

    # ── Approach 2: Isolation Forest ─────────────────────
    iforest = IsolationForestDetector(contamination=0.35)
    iforest.fit(X_train)
    y_pred_if = iforest.predict(X_test)
    evaluate("Isolation Forest Detector", y_test, y_pred_if)

    # ── Quick demo predictions ───────────────────────────
    print("\n═══ Quick demo predictions ═══")
    sample_malignant = np.array(
        [[0.5, 1, 1, 0.8, 0.5, 0.5, 0.7, 1, 0.1]]
    )
    sample_benign = np.array(
        [[0.5, 0.4, 0.5, 0.1, 0.8, 0.1, 0.3, 0.6, 0.1]]
    )

    for name, det in [("Gaussian", gauss),
                       ("IsolationForest", iforest)]:
        m = det.predict(sample_malignant)[0]
        b = det.predict(sample_benign)[0]
        print(
            f"  {name:16s}  malignant→"
            f"{'ANOMALY' if m else 'normal':8s}  "
            f"benign→{'ANOMALY' if b else 'normal'}"
        )


if __name__ == "__main__":
    main()
