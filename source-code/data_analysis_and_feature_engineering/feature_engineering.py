"""
Feature Engineering example using the California Housing dataset.

Demonstrates: handling missing data, encoding categorical features,
feature scaling, and feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def load_and_engineer():
    """Load data and create engineered features."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target

    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}\n")

    # --- 1. Create new features from existing ones ---
    df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]
    df["PopPerHousehold"] = df["Population"] / df["HouseAge"]

    print("=== New Engineered Features ===")
    print(df[["RoomsPerHousehold", "BedroomRatio",
              "PopPerHousehold"]].describe().round(3).to_string())
    print()

    # --- 2. Create a categorical feature and encode it ---
    # Bin latitude into regions (North/Central/South California)
    df["Region"] = pd.cut(
        df["Latitude"],
        bins=[32, 35, 38, 42],
        labels=["South", "Central", "North"]
    )
    print("=== Region Distribution ===")
    print(df["Region"].value_counts().sort_index().to_string())
    print()

    # One-hot encode the categorical feature
    region_dummies = pd.get_dummies(df["Region"], prefix="Region", dtype=int)
    df = pd.concat([df, region_dummies], axis=1)
    df = df.drop(columns=["Region"])

    # --- 3. Simulate and handle missing data ---
    rng = np.random.default_rng(42)
    mask = rng.random(len(df)) < 0.05  # ~5% missing
    df.loc[mask, "PopPerHousehold"] = np.nan

    n_missing = df["PopPerHousehold"].isnull().sum()
    print(f"=== Handling Missing Data ===")
    print(f"Introduced {n_missing} missing values in PopPerHousehold")

    # Fill with median (robust to outliers)
    median_val = df["PopPerHousehold"].median()
    df["PopPerHousehold"] = df["PopPerHousehold"].fillna(median_val)
    print(f"Filled with median: {median_val:.3f}")
    print(f"Remaining missing: {df['PopPerHousehold'].isnull().sum()}\n")

    # --- 4. Feature scaling comparison ---
    print("=== Feature Scaling Impact ===")
    features_to_show = ["MedInc", "AveRooms", "Population"]
    print("Before scaling:")
    for f in features_to_show:
        print(f"  {f:15s}  mean={df[f].mean():10.2f}  std={df[f].std():10.2f}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features_to_show])
    scaled_df = pd.DataFrame(scaled, columns=features_to_show)

    print("After StandardScaler:")
    for f in features_to_show:
        print(f"  {f:15s}  mean={scaled_df[f].mean():10.4f}"
              f"  std={scaled_df[f].std():10.4f}")
    print()

    return df


def compare_models(df):
    """Compare model performance with and without engineered features."""
    original_features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ]
    engineered_features = original_features + [
        "RoomsPerHousehold", "BedroomRatio", "PopPerHousehold",
        "Region_South", "Region_Central", "Region_North",
    ]

    y = df["MedHouseVal"]

    print("=== Model Comparison: Original vs. Engineered Features ===")
    for label, features in [("Original (8 features)", original_features),
                            ("Engineered (14 features)", engineered_features)]:
        X = df[features]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        r2 = r2_score(y_test, y_pred)
        print(f"  {label:30s}  R² = {r2:.4f}")
    print()


if __name__ == "__main__":
    df = load_and_engineer()
    compare_models(df)
