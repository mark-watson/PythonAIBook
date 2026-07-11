import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from anomaly_detection import (
    GaussianAnomalyDetector,
    IsolationForestDetector,
    evaluate,
)

# See src/anomaly_detection/detectors.py for the reason we use bare np.ndarray.
type FloatArray = np.ndarray
type IntArray = np.ndarray


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_blob(
    n: int, n_features: int, center: float, spread: float, seed: int
) -> FloatArray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=center, scale=spread, size=(n, n_features))


@pytest.fixture
def train_data() -> FloatArray:
    return _make_blob(n=200, n_features=5, center=0.5, spread=0.1, seed=1)


@pytest.fixture
def eval_data() -> tuple[FloatArray, IntArray]:
    normal = _make_blob(n=80, n_features=5, center=0.5, spread=0.1, seed=2)
    anomalies = _make_blob(n=20, n_features=5, center=3.0, spread=0.1, seed=3)
    X = np.vstack([normal, anomalies])
    y = np.concatenate([np.zeros(80, dtype=int), np.ones(20, dtype=int)])
    return X, y


# ── GaussianAnomalyDetector ───────────────────────────────────────────────────


def test_gaussian_fit_populates_stats(train_data: FloatArray) -> None:
    det = GaussianAnomalyDetector()
    det.fit(train_data)
    assert det.mu is not None
    assert det.sigma_sq is not None
    assert det.mu.shape == (5,)
    assert det.sigma_sq.shape == (5,)
    assert (det.sigma_sq > 0).all()


def test_gaussian_flags_anomalies(
    train_data: FloatArray, eval_data: tuple[FloatArray, IntArray]
) -> None:
    X_cv, y_cv = eval_data
    det = GaussianAnomalyDetector()
    det.fit(train_data, y_cv=y_cv, X_cv=X_cv)
    preds = det.predict(X_cv)
    # The blob-based anomalies are far from the normal distribution,
    # so recall on the anomaly class should be high.
    recall = float(np.sum(preds & (y_cv == 1))) / float(np.sum(y_cv == 1))
    assert recall > 0.9


def test_gaussian_score_matches_probability(train_data: FloatArray) -> None:
    det = GaussianAnomalyDetector()
    det.fit(train_data)
    scores = det.score(train_data)
    assert scores.shape == (train_data.shape[0],)
    assert (scores > 0).all()


# ── IsolationForestDetector ───────────────────────────────────────────────────


def test_isolation_forest_flags_anomalies(
    train_data: FloatArray, eval_data: tuple[FloatArray, IntArray]
) -> None:
    X_cv, y_cv = eval_data
    det = IsolationForestDetector(contamination=0.2)
    det.fit(train_data)
    preds = det.predict(X_cv)
    recall = float(np.sum(preds & (y_cv == 1))) / float(np.sum(y_cv == 1))
    assert recall > 0.9


def test_isolation_forest_predict_returns_boolean(train_data: FloatArray) -> None:
    det = IsolationForestDetector()
    det.fit(train_data)
    preds = det.predict(train_data)
    assert preds.dtype == np.bool_
    assert preds.shape == (train_data.shape[0],)


# ── evaluate() helper ─────────────────────────────────────────────────────────


def test_evaluate_returns_perfect_f1_when_predictions_match() -> None:
    y_true: IntArray = np.array([0, 1, 0, 1, 1], dtype=int)
    y_pred = y_true.astype(bool)
    f1 = evaluate("perfect", y_true, y_pred)
    assert f1 == pytest.approx(1.0)


def test_evaluate_returns_zero_f1_when_all_wrong() -> None:
    y_true: IntArray = np.array([0, 1, 0, 1], dtype=int)
    y_pred = (1 - y_true).astype(bool)
    f1 = evaluate("all wrong", y_true, y_pred)
    assert f1 == pytest.approx(0.0)


# ── property-based tests ──────────────────────────────────────────────────────


@given(
    st.integers(min_value=20, max_value=200),
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=20, deadline=None)
def test_gaussian_score_shape(n: int, n_features: int, seed: int) -> None:
    X = _make_blob(n=n, n_features=n_features, center=0.0, spread=1.0, seed=seed)
    det = GaussianAnomalyDetector()
    det.fit(X)
    scores = det.score(X)
    assert scores.shape == (n,)
