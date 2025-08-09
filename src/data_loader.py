import os
import ssl

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_iris():
    """Load, preprocess, and save the Iris dataset."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load dataset into DataFrame
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Save raw dataset
    X.assign(target=y).to_csv("data/raw/iris.csv", index=False)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed train/test
    pd.DataFrame(X_train_scaled, columns=X.columns).assign(target=y_train).to_csv(
        "data/processed/iris_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).assign(target=y_test).to_csv(
        "data/processed/iris_test.csv", index=False
    )

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Iris data processed and saved.")
    return X_train_scaled, X_test_scaled, y_train, y_test


def load_and_preprocess_housing():
    """Load, preprocess, and save the California Housing dataset."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Fix SSL issue for dataset download
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load dataset into DataFrame
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target

    # Save raw dataset
    X.assign(target=y).to_csv("data/raw/housing.csv", index=False)

    # Clean / fill NAs
    X = X.fillna(X.median())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed train/test
    pd.DataFrame(X_train_scaled, columns=X.columns).assign(target=y_train).to_csv(
        "data/processed/housing_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).assign(target=y_test).to_csv(
        "data/processed/housing_test.csv", index=False
    )

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Housing data processed and saved.")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # For DVC, run both datasets in one go
    load_and_preprocess_iris()
    load_and_preprocess_housing()
