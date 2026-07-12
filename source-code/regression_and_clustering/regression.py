"""
Regression example using the California Housing dataset.

Demonstrates Linear Regression and Polynomial Regression
with scikit-learn.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_housing_data():
    """Load the California Housing dataset into a DataFrame."""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target
    return df


def run_linear_regression(df):
    """Train and evaluate a Linear Regression model."""
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("=== Linear Regression Results ===")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
    print()

    # Show feature importance via coefficients
    coefs = pd.Series(model.coef_, index=X.columns)
    print("Feature coefficients (scaled):")
    for name, val in coefs.items():
        print(f"  {name:20s} {val:+.4f}")
    print()


def run_polynomial_regression(df):
    """Train a Polynomial Regression model using two features."""
    # Use only MedInc and AveRooms to keep it interpretable
    X = df[["MedInc", "AveRooms"]].copy()
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    print("=== Polynomial Regression (degree=2, 2 features) ===")
    print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  R²:   {r2_score(y_test, y_pred):.4f}")
    print()
    print(f"  Polynomial feature names: {poly.get_feature_names_out().tolist()}")
    print()


if __name__ == "__main__":
    df = load_housing_data()
    print(f"Dataset shape: {df.shape}")
    print(
        f"Target range: {df['MedHouseVal'].min():.2f} - "
        f"{df['MedHouseVal'].max():.2f} "
        f"(units: $100,000s)\n"
    )
    run_linear_regression(df)
    run_polynomial_regression(df)
