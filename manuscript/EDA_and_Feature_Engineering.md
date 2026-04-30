# Exploratory Data Analysis and Feature Engineering

Before training any machine learning model, you need to understand your data. **Exploratory Data Analysis (EDA)** is the process of examining a dataset to summarize its main characteristics, find patterns, detect anomalies, and check assumptions. **Feature engineering** is the art of creating new input variables — or transforming existing ones — to improve model performance.

These steps often make the difference between a mediocre model and a good one. As the saying goes: "garbage in, garbage out."

The requirements for this chapter are:

```bash
uv pip install scikit-learn pandas numpy
```

The examples for this chapter are in the directory **source-code/data_analysis_and_feature_engineering**.

We continue using the California Housing dataset from the previous chapter.

## Exploratory Data Analysis

### Loading and Inspecting the Data

The first thing to do with any dataset is to understand its shape, types, and basic statistics:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.dtypes)
```

Running our **eda.py** script gives us:

```bash
$ python eda.py
=== Dataset Overview ===
Shape: 20640 rows × 9 columns

Column types:
MedInc         float64
HouseAge       float64
AveRooms       float64
AveBedrms      float64
Population     float64
AveOccup       float64
Latitude       float64
Longitude      float64
MedHouseVal    float64
```

All columns are floating point. Next, summary statistics:

```bash
=== Summary Statistics ===
         MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup
count  20640.00  20640.00  20640.00   20640.00    20640.00  20640.00
mean       3.87     28.64      5.43       1.10     1425.48      3.07
std        1.90     12.59      2.47       0.47     1132.46     10.39
min        0.50      1.00      0.85       0.33        3.00      0.69
max       15.00     52.00    141.91      34.07    35682.00   1243.33
```

Notice the wide range differences: **Population** ranges from 3 to 35,682 while **AveBedrms** ranges from 0.33 to 34.07. This tells us we will need feature scaling before training most models.

Also notice the extreme maximum values for **AveRooms** (141.91) and **AveOccup** (1,243.33) — these are likely outliers or data quality issues.

### Checking for Missing Values

Missing data can silently break your models or introduce bias. Always check:

```bash
=== Missing Values ===
No missing values found.
```

This dataset is clean, but real-world data rarely is. We will practice handling missing values in the feature engineering section below.

### Correlation Analysis

Understanding which features correlate with the target helps guide feature selection and engineering:

```bash
=== Correlation with MedHouseVal ===
  MedInc                +0.6881
  AveRooms              +0.1519
  Latitude              -0.1442
  HouseAge              +0.1056
  AveBedrms             -0.0467
  Longitude             -0.0460
  Population            -0.0246
  AveOccup              -0.0237
```

**MedInc** (median income) stands out with a correlation of +0.69 — by far the strongest predictor. This aligns with what we saw from the regression coefficients in the previous chapter. The other features have relatively weak linear correlations, suggesting that non-linear relationships or feature combinations might be more informative.

### Outlier Detection

The IQR (Interquartile Range) method flags values that fall more than 1.5 × IQR below Q1 or above Q3:

```bash
=== Outlier Counts (IQR method) ===
  MedInc                  681 outliers (3.3%)
  AveRooms                511 outliers (2.5%)
  AveBedrms              1424 outliers (6.9%)
  Population             1196 outliers (5.8%)
  AveOccup                711 outliers (3.4%)
  MedHouseVal            1071 outliers (5.2%)
```

Nearly 7% of **AveBedrms** values are outliers. In practice, you would investigate whether these are genuine extreme values or data errors, and decide whether to clip, transform, or remove them depending on your use case.


## Feature Engineering

Feature engineering is where domain knowledge meets data science. By creating new features that better represent the underlying patterns, we can significantly improve model performance — sometimes more than choosing a fancier algorithm.

### Creating New Features

We can derive meaningful features by combining existing ones:

```python
df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
df["PopPerHousehold"] = df["Population"] / df["HouseAge"]
```

- **RoomsPerHousehold**: a proxy for house size relative to occupancy.
- **BedroomRatio**: what fraction of rooms are bedrooms (a measure of house layout).
- **PopPerHousehold**: population growth rate proxy (newer areas with high population).

### Encoding Categorical Features

Many real-world datasets contain categorical variables (e.g., "color", "region", "type"). Most ML algorithms require numerical inputs, so we need to encode these.

In our example, we create a categorical feature by binning latitude into California regions, then **one-hot encode** it:

```python
df["Region"] = pd.cut(
    df["Latitude"],
    bins=[32, 35, 38, 42],
    labels=["South", "Central", "North"]
)

region_dummies = pd.get_dummies(df["Region"], prefix="Region", dtype=int)
df = pd.concat([df, region_dummies], axis=1)
```

```bash
=== Region Distribution ===
Region
South      11294
Central     6331
North       3015
```

One-hot encoding creates a separate binary column for each category (**Region_South**, **Region_Central**, **Region_North**). This avoids imposing a false numerical ordering on the categories.

### Handling Missing Data

Real datasets almost always have missing values. Common strategies include:

- **Drop rows**: simple but loses data.
- **Fill with mean/median**: preserves dataset size; median is more robust to outliers.
- **Fill with a model prediction**: more sophisticated but adds complexity.

We demonstrate median imputation:

```python
# Simulate 5% missing values
rng = np.random.default_rng(42)
mask = rng.random(len(df)) < 0.05
df.loc[mask, "PopPerHousehold"] = np.nan

# Fill with median
median_val = df["PopPerHousehold"].median()
df["PopPerHousehold"] = df["PopPerHousehold"].fillna(median_val)
```

```bash
=== Handling Missing Data ===
Introduced 1028 missing values in PopPerHousehold
Filled with median: 41.833
Remaining missing: 0
```

### Feature Scaling

Features on vastly different scales cause problems for distance-based algorithms (K-NN, K-Means) and gradient-based optimizers. **StandardScaler** transforms each feature to have zero mean and unit variance:

```bash
=== Feature Scaling Impact ===
Before scaling:
  MedInc           mean=      3.87  std=      1.90
  AveRooms         mean=      5.43  std=      2.47
  Population       mean=   1425.48  std=   1132.46
After StandardScaler:
  MedInc           mean=    0.0000  std=    1.0000
  AveRooms         mean=    0.0000  std=    1.0000
  Population       mean=   -0.0000  std=    1.0000
```

After scaling, all features are on the same footing. Remember to always fit the scaler on training data only and apply it to both train and test sets to prevent data leakage.

### Measuring the Impact

The ultimate test of feature engineering is whether it improves model performance. We compare a Linear Regression model with the original 8 features against one with our 14 engineered features:

```bash
=== Model Comparison: Original vs. Engineered Features ===
  Original (8 features)           R² = 0.5758
  Engineered (14 features)        R² = 0.6622
```

Our engineered features improved R² from 0.58 to 0.66 — a **15% improvement** in explained variance, using the exact same algorithm. This demonstrates why feature engineering is often more valuable than model selection for improving results.


## EDA and Feature Engineering Wrap-up

In this chapter we covered the essential data preparation skills that precede model training:

- **EDA** helps you understand your data through summary statistics, correlation analysis, and outlier detection. Never skip this step.
- **Feature engineering** transforms raw data into more informative inputs: creating derived features, encoding categories, handling missing values, and scaling.
- The payoff is real: our engineered features produced a 15% improvement in model performance with zero algorithm changes.

These techniques apply to every machine learning project, whether you are using classic algorithms from scikit-learn or deep learning frameworks. In the next part of this book, we move into deep learning.

