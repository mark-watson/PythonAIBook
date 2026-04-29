"""
Exploratory Data Analysis example using the California Housing dataset.

Demonstrates: loading data, summary statistics, correlation analysis,
and identifying outliers.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def main():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target

    # Basic info
    print("=== Dataset Overview ===")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print("Column types:")
    print(df.dtypes.to_string())
    print()

    # Summary statistics
    print("=== Summary Statistics ===")
    print(df.describe().round(2).to_string())
    print()

    # Check for missing values
    missing = df.isnull().sum()
    print("=== Missing Values ===")
    if missing.sum() == 0:
        print("No missing values found.\n")
    else:
        print(missing[missing > 0].to_string())
        print()

    # Correlation with target
    print("=== Correlation with MedHouseVal ===")
    corr = df.corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal")
    corr_sorted = corr.abs().sort_values(ascending=False)
    for feature in corr_sorted.index:
        print(f"  {feature:20s}  {corr[feature]:+.4f}")
    print()

    # Outlier detection using IQR method
    print("=== Outlier Counts (IQR method) ===")
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            pct = 100 * n_outliers / len(df)
            print(f"  {col:20s}  {n_outliers:5d} outliers ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
