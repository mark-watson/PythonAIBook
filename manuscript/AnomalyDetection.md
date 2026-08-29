# Anomaly Detection

Anomaly detection is the task of identifying data points that deviate significantly from the expected pattern. Unlike classification, where we have balanced training examples for each class, anomaly detection is designed for situations where "normal" examples vastly outnumber "anomalous" ones, often by 100:1 or more. Fraud detection, network intrusion monitoring, manufacturing quality control, and medical diagnosis are all domains where anomaly detection excels.

The key insight is that we can build a model of what "normal" looks like and then flag anything that doesn't fit. This chapter implements two complementary approaches:

1. **Gaussian Statistical Detector**: a from-scratch implementation that models each feature with a Gaussian distribution and uses a tunable probability threshold.
2. **Isolation Forest**: the current industry-standard tree-based algorithm from scikit-learn that requires no labeled data at all.

By comparing both approaches on the same dataset, we will see the tradeoffs between supervised tuning and fully unsupervised detection.

The examples for this chapter are in the directory **source-code/anomaly_detection**. The project is a modern `uv`-managed Python package (with `pyrefly` strict typing, `ruff` formatting, `beartype`/`typeguard` runtime type enforcement, and Claude Code hooks). From the project directory, one command syncs the environment and installs all runtime and dev dependencies (numpy, matplotlib, scikit-learn, pandas, pytest, hypothesis, etc.):

```bash
uv sync
```

The `justfile` provides the developer workflow: `just check` runs format-check + lint + typecheck + tests, `just run` runs the Wisconsin example, and `just test` runs the unit and property-based tests. See the project `README.md` for the full setup guide.


{width: "80%"}
![Architecture diagram for the Anomaly Detection example](FIG_anomaly_detection.jpg)


## The Wisconsin Breast Cancer Dataset

We reuse the Wisconsin Diagnostic Breast Cancer dataset from earlier chapters, this time treating malignant samples as anomalies rather than a classification target. The dataset contains 648 samples with 9 features measuring cell characteristics (clump thickness, uniformity of cell size and shape, marginal adhesion, etc.) and a class label: 2 for benign, 4 for malignant.

Roughly 35% of samples are malignant, a higher anomaly rate than most real-world problems, but useful for demonstrating the techniques with enough anomalies to evaluate precision and recall meaningfully.


## Data Preprocessing

Good anomaly detection requires careful preprocessing. Many statistical detectors assume features follow an approximately Gaussian (bell-curve) distribution, so we apply a log-transform followed by per-row min–max scaling to push the data closer to that assumption:

```python
raw = np.genfromtxt(DATA_PATH, delimiter=",")
X_raw = raw[:, :9] * 0.1           # scale to [0, 1]

# log-transform to approximate Gaussian shape
X_log = np.log(X_raw + 1.2)
row_min = X_log.min(axis=1, keepdims=True)
row_max = X_log.max(axis=1, keepdims=True)
X = (X_log - row_min) / (row_max - row_min + 1e-10)

# Target: map class 2 → 0 (normal), class 4 → 1 (anomaly)
y = ((raw[:, 9] - 2) * 0.5).astype(int)
```

We then split the data three ways: 60% training, 20% cross-validation, 20% test. Crucially, the training set is built from mostly normal (benign) examples, with only about 10% anomalies allowed through. This mimics the real-world scenario where we train on data that is overwhelmingly normal:

```python
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
```

The script also generates a 3×3 grid of per-feature histograms colour-coded by class, saved to `histograms.png`. Examining these distributions is always a good first step; features where the normal and anomaly distributions overlap heavily will be harder for any detector to leverage.


## Approach 1: Gaussian Statistical Detector

The Gaussian approach is mathematically elegant and gives deep insight into *why* anomaly detection works. The idea comes from Andrew Ng's machine learning course and is a staple of introductory ML curricula.

### The Algorithm

For each feature, we compute the mean (μ) and variance (σ²) from the training data. Given a new observation **x**, we compute the probability of each feature under its Gaussian distribution and average the results:

```$
p(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^{n} \frac{1}{\sqrt{2\pi}\,\sigma_i} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right)
```

If this aggregate probability falls below a threshold **epsilon** (ε), the observation is flagged as an anomaly.

### Implementation

The `GaussianAnomalyDetector` class in **src/anomaly_detection/detectors.py** implements this in about 50 lines of Python:

```python
class GaussianAnomalyDetector:

    def __init__(self):
        self.mu = None
        self.sigma_sq = None
        self.epsilon = 0.02

    def fit(self, X_train, y_cv=None, X_cv=None):
        self.mu = X_train.mean(axis=0)
        self._fit_sigma(X_train)
        if X_cv is not None and y_cv is not None:
            self._tune_epsilon(X_cv, y_cv)

    def _fit_sigma(self, X):
        self.sigma_sq = (
            np.sum((X - self.mu) ** 2, axis=0)
            / X.shape[0]
        )
        self.sigma_sq = np.maximum(self.sigma_sq, 1e-10)

    def _probability(self, X):
        exponent = (
            -((X - self.mu) ** 2) / (2.0 * self.sigma_sq)
        )
        per_feature = (
            1.0 / (SQRT_2_PI * np.sqrt(self.sigma_sq))
        ) * np.exp(exponent)
        return per_feature.mean(axis=1)

    def predict(self, X):
        return self._probability(X) < self.epsilon
```

A few things to note:

- The variance is computed as the mean of squared deviations from the mean, the standard formula for population variance.
- We guard against zero variance with `np.maximum(self.sigma_sq, 1e-10)` to avoid division by zero in features with constant values.
- The `predict` method returns `True` for anomalies: observations whose probability falls below epsilon.

### Tuning Epsilon

The threshold ε is a hyperparameter that controls the sensitivity of the detector. Too low, and anomalies slip through (low recall). Too high, and normal observations get flagged (low precision).

We tune epsilon on the cross-validation set by sweeping through a range of candidate values and selecting the one that minimises classification errors:

```python
def _tune_epsilon(self, X_cv, y_cv):
    best_err, best_eps = 1e10, self.epsilon
    for i in range(200):
        eps = 0.001 + 0.005 * i
        preds = self._probability(X_cv) < eps
        err = np.sum(preds != y_cv)
        if err <= best_err:
            best_err, best_eps = err, eps
    self.epsilon = best_eps
```

Note the `<=` comparison: when multiple epsilon values produce the same error count, we prefer the highest one. This gives a more generous threshold that is less likely to overfit to the cross-validation set.


## Approach 2: Isolation Forest

Isolation Forest, introduced by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou in 2008, is the industry-standard baseline for anomaly detection on tabular data. The core idea is beautifully simple: anomalies are *easier to isolate* than normal points.

### How It Works

The algorithm builds an ensemble of random trees (similar to Random Forest, but without labels). Each tree recursively partitions the data by choosing a random feature and a random split point. Normal observations, which are surrounded by similar points, require many splits to isolate. Anomalies, which are few and different, get isolated in just a few splits.

The **anomaly score** is based on the average path length from the root to the leaf across all trees. Shorter paths mean the observation was easy to isolate, and therefore more anomalous.

### Implementation

Because scikit-learn provides a high-quality implementation, our wrapper is just a thin adapter that matches the interface of the Gaussian detector:

```python
class IsolationForestDetector:

    def __init__(self, contamination=0.1, n_estimators=200,
                 random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X_train, y_cv=None, X_cv=None):
        self.model.fit(X_train)

    def predict(self, X):
        return self.model.predict(X) == -1
```

The key hyperparameter is `contamination`, the expected proportion of anomalies in the training data. Setting this correctly is critical: too low and the model misses anomalies; too high and it over-flags normal data. In our example we set it to 0.35 to match the dataset's actual anomaly rate.

Unlike the Gaussian detector, Isolation Forest is **fully unsupervised**: it does not use the cross-validation labels at all. This is both its strength (works without any labels) and its weakness (cannot tune a decision boundary to match known anomaly patterns).


## Running the Example

Running the complete example (either `just run` or the underlying `uv run` invocation):

```bash
$ just run
uv run python -m anomaly_detection.wisconsin
Training examples  : 264
Cross-val examples : 129
Test examples      : 131

Histograms saved to histograms.png
  Gaussian detector — best epsilon = 0.9960  (CV errors: 16)

──────────────────────────────────────────────────
  Gaussian Statistical Detector
──────────────────────────────────────────────────
  Precision : 0.9024
  Recall    : 0.8043
  F1        : 0.8506

              precision    recall  f1-score   support

      normal       0.90      0.95      0.93        85
     anomaly       0.90      0.80      0.85        46

    accuracy                           0.90       131
   macro avg       0.90      0.88      0.89       131
  weighted avg       0.90      0.90      0.90       131


──────────────────────────────────────────────────
  Isolation Forest Detector
──────────────────────────────────────────────────
  Precision : 0.5897
  Recall    : 1.0000
  F1        : 0.7419

              precision    recall  f1-score   support

      normal       1.00      0.62      0.77        85
     anomaly       0.59      1.00      0.74        46

    accuracy                           0.76       131
   macro avg       0.79      0.81      0.76       131
  weighted avg       0.86      0.76      0.76       131
```


## Interpreting the Results

The results reveal an important lesson about the tradeoff between supervised and unsupervised approaches:

**Gaussian Detector (F1 = 0.85)**: Because it uses cross-validation labels to tune epsilon, it achieves excellent balance: 90% precision with 80% recall. It correctly classifies 90% of all test samples.

**Isolation Forest (F1 = 0.74)**: It catches *every* anomaly (100% recall) but at the cost of many false positives (59% precision). It flags 38% of normal samples as anomalous. This is a common characteristic of unsupervised detectors: they err on the side of caution.

The takeaway: **when you have even a small set of labeled anomalies for tuning, a simpler statistical model can outperform a more sophisticated unsupervised one.** In practice, many production systems use a hybrid approach: an unsupervised detector for initial screening, followed by a tuned model (or human review) for the final decision.

### Evaluation Metrics for Anomaly Detection

Standard accuracy is misleading for anomaly detection because the classes are imbalanced. If 95% of data is normal, a model that always predicts "normal" achieves 95% accuracy while catching zero anomalies. Instead, focus on:

- **Precision**: Of the observations flagged as anomalies, how many actually are? High precision means few false alarms.
- **Recall**: Of all true anomalies, how many did we catch? High recall means few missed anomalies.
- **F1 Score**: The harmonic mean of precision and recall, a single number that balances both concerns.

The right balance depends on your domain. In fraud detection, missing a fraud (low recall) is costly, so you tolerate more false positives. In manufacturing, false alarms that shut down a production line (low precision) are costly, so you set a higher threshold.


## Anomaly Detection Wrap-up

In this chapter we implemented two complementary approaches to anomaly detection:

- The **Gaussian statistical detector** gives us mathematical transparency: we can inspect the learned μ and σ² values per feature and understand exactly why an observation was flagged. The epsilon threshold provides a single, interpretable knob for controlling sensitivity.
- The **Isolation Forest** requires no labels and scales well to high-dimensional data. It is the recommended starting point for any new anomaly detection project where labeled anomalies are unavailable.

Both approaches are widely used in practice, and understanding their tradeoffs (supervised tuning vs. unsupervised convenience, interpretability vs. scalability) is essential for any practitioner working with anomaly detection problems.


## Optional Practice Problems

Here are some optional practice problems to help you deepen your understanding of anomaly detection and extend the code implemented in this chapter.

### Problem 1 (Easy): Independent Gaussian Log-Likelihood
The current implementation of the [GaussianAnomalyDetector](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/detectors.py#L42) computes the arithmetic mean of the individual feature probabilities to determine the overall probability of a sample:
```python
return per_feature.mean(axis=1)
```
If we assume that the features are conditionally independent given the class, the joint probability of a sample is the product of its feature probabilities:

```$
p(\mathbf{x}) = \prod_{i=1}^{n} p(x_i)
```

Because multiplying small probabilities leads to numerical underflow, we typically compute the sum of log-probabilities instead:

```$
\log p(\mathbf{x}) = \sum_{i=1}^{n} \log p(x_i)
```

**Task:**
1. Modify the `_probability` method of [GaussianAnomalyDetector](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/detectors.py#L42) to compute the joint log-probability of each sample. (Hint: compute the natural logarithm of individual feature probability densities, sum them across axis 1, and return this log-probability).
2. Update the `_tune_epsilon` and `predict` methods accordingly. Note that since log-probabilities are negative numbers, `epsilon` will also be negative, and a sample is an anomaly when its log-probability is *lower* (more negative) than `epsilon`. You will need to adjust the range of candidate thresholds swept in `_tune_epsilon`.
3. Tune `epsilon` on the cross-validation set and evaluate the new log-likelihood detector on the test set.
4. Compare the resulting Precision, Recall, and F1 scores with those of the original arithmetic-mean implementation.

### Problem 2 (Medium): Semi-Supervised Hyperparameter Tuning for Isolation Forest
Currently, our [IsolationForestDetector](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/detectors.py#L132) wrapper is fully unsupervised and uses a hardcoded `contamination=0.35` hyperparameter:
```python
iforest = IsolationForestDetector(contamination=0.35)
```
In real-world settings, we often have a small labeled dataset (e.g., our cross-validation set) that we can use to tune hyperparameters, even if the model itself is trained in an unsupervised manner on the training set.

**Task:**
1. Extend the `fit` method of [IsolationForestDetector](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/detectors.py#L132) to accept optional validation data `X_cv` and `y_cv`.
2. Implement a grid search within `fit` that sweeps over a range of hyperparameters when validation data is provided:
   - `contamination`: from `0.05` to `0.50` in steps of `0.05`
   - `n_estimators`: `[50, 100, 200, 300]`
3. For each combination, fit the `IsolationForest` model on `X_train`, predict on `X_cv`, and calculate the validation F1 score.
4. Store the best-performing hyperparameters and fit the final model using them.
5. Integrate this tuning step in [wisconsin_anomaly.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/wisconsin.py), evaluate the optimized model on the test set, and compare the results to the default model. How much did tuning improve the Precision and F1 score?

### Problem 3 (Hard): Density-based Detection with Local Outlier Factor (LOF) and PCA
While Isolation Forest isolates anomalies using random partitioning, distance- or density-based methods like **Local Outlier Factor (LOF)** identify anomalies by comparing the local density of a point to that of its neighbors. In high-dimensional spaces, these density metrics can degrade due to the "curse of dimensionality."

**Task:**
1. Implement a new class `LOFDetector` in [anomaly_detection.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/detectors.py) wrapping scikit-learn's `LocalOutlierFactor`. Set `novelty=True` in the constructor so you can call `fit` on training data and `predict`/`score` on test data.
2. Write a Python script that applies Principal Component Analysis (PCA) to reduce the 9-dimensional cancer dataset to 2 dimensions.
3. Train all three detectors ([GaussianAnomalyDetector](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/detectors.py#L42), [IsolationForestDetector](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/anomaly_detection/src/anomaly_detection/detectors.py#L132), and your new `LOFDetector`) on:
   - The original 9-dimensional space.
   - The 2-dimensional PCA space.
4. Evaluate and compare the Precision, Recall, and F1 scores for all configurations on the test set.
5. Using `matplotlib`, plot the 2D PCA-reduced test set. Create subplots showing:
   - The true class labels (normal vs. anomaly).
   - The predictions of each detector (correctly identified normal points, true positives, false positives, and false negatives).
   - Analyze which patterns each detector struggled with or excelled at resolving.
