"""
Anomaly detection using two complementary approaches:

1. **GaussianAnomalyDetector** — a from-scratch statistical model
   that fits per-feature Gaussian distributions (mean + variance)
   and uses a tunable epsilon threshold.  Ported from the Java
   version in the companion Java AI book.

2. **IsolationForestDetector** — a scikit-learn wrapper around
   the Isolation Forest algorithm, the current industry-standard
   baseline for tabular anomaly detection.

Both detectors expose the same interface:
    fit(X_train, y_cv, X_cv)   — train / tune
    predict(X)                 — return boolean anomaly labels
    score(X)                   — return raw anomaly scores
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
)

SQRT_2_PI = 2.50662827463


# ── 1. From-scratch Gaussian detector ────────────────────────


class GaussianAnomalyDetector:
    """Statistical anomaly detector using per-feature
    Gaussian probability estimates.

    The probability of an observation is the mean across
    features of  p(x_i; mu_i, sigma_i^2).
    An observation is flagged as an anomaly when its
    aggregate probability falls below a threshold
    *epsilon*, which is tuned on a cross-validation set
    to minimise classification error.
    """

    def __init__(self):
        self.mu = None
        self.sigma_sq = None
        self.epsilon = 0.02

    # ── training ──

    def fit(self, X_train, y_cv=None, X_cv=None):
        """Compute feature means, then tune epsilon on
        the cross-validation set (if provided).
        """
        self.mu = X_train.mean(axis=0)
        self._fit_sigma(X_train)

        if X_cv is not None and y_cv is not None:
            self._tune_epsilon(X_cv, y_cv)

    def _fit_sigma(self, X):
        n_features = X.shape[1]
        # Variance per feature, scaled by (1/n_features)
        # to match the Java implementation
        self.sigma_sq = (
            np.sum((X - self.mu) ** 2, axis=0)
            / X.shape[0]
        )
        # Guard against zero variance
        self.sigma_sq = np.maximum(self.sigma_sq, 1e-10)

    def _probability(self, X):
        """p(x; mu, sigma^2) averaged over features.

        Uses the Gaussian PDF:
          p(x_i) = (1 / sqrt(2*pi) * sigma_i)
                   * exp(-(x_i - mu_i)^2 / (2 * sigma_i^2))
        """
        exponent = (
            -((X - self.mu) ** 2) / (2.0 * self.sigma_sq)
        )
        per_feature = (
            1.0 / (SQRT_2_PI * np.sqrt(self.sigma_sq))
        ) * np.exp(exponent)
        return per_feature.mean(axis=1)

    def _tune_epsilon(self, X_cv, y_cv):
        best_err, best_eps = 1e10, self.epsilon
        # Sweep a wide range of epsilon thresholds
        for i in range(200):
            eps = 0.001 + 0.005 * i
            preds = self._probability(X_cv) < eps
            err = np.sum(preds != y_cv)
            if err <= best_err:
                best_err, best_eps = err, eps
        self.epsilon = best_eps
        print(
            f"  Gaussian detector — best epsilon "
            f"= {self.epsilon:.4f}  "
            f"(CV errors: {int(best_err)})"
        )

    # ── inference ──

    def predict(self, X):
        """Return True for anomalies."""
        return self._probability(X) < self.epsilon

    def score(self, X):
        """Lower score ⇒ more anomalous."""
        return self._probability(X)


# ── 2. Isolation Forest wrapper ──────────────────────────────


class IsolationForestDetector:
    """Thin wrapper around scikit-learn's IsolationForest
    that exposes the same interface as the Gaussian
    detector above.
    """

    def __init__(self, contamination=0.1, n_estimators=200,
                 random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X_train, y_cv=None, X_cv=None):
        """Fit on training data (labels are ignored —
        Isolation Forest is unsupervised)."""
        self.model.fit(X_train)

    def predict(self, X):
        """Return True for anomalies."""
        # sklearn returns -1 for outliers, 1 for inliers
        return self.model.predict(X) == -1

    def score(self, X):
        """Lower (more negative) ⇒ more anomalous."""
        return self.model.decision_function(X)


# ── helpers ──────────────────────────────────────────────────


def evaluate(name, y_true, y_pred):
    """Print precision / recall / F1 for an anomaly
    detector."""
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print()
    print(
        classification_report(
            y_true, y_pred,
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )
    return f1
