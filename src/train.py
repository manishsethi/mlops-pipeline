# src/train.py

import argparse
import json
import joblib
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from src.data_loader import (
    load_and_preprocess_iris,
    load_and_preprocess_housing
)
from src.mlflow_config import setup_mlflow

def train_classification():
    X_train, X_test, y_train, y_test = load_and_preprocess_iris()
    models = {
        "logistic_regression": LogisticRegression(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    best_score, best_name, best_model = 0, None, None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_test[:5]

            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=f"iris_{name}",
                signature=signature,
                input_example=input_example
            )

            if acc > best_score:
                best_score, best_name, best_model = acc, name, model

    return best_name, best_model, {"accuracy": best_score, "feature_count": X_train.shape[1]}

def train_regression():
    X_train, X_test, y_train, y_test = load_and_preprocess_housing()
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    best_score, best_name, best_model = float("inf"), None, None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            signature = infer_signature(X_train, model.predict(X_train))
            input_example = X_test[:5]

            mlflow.log_params(model.get_params())
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=f"housing_{name}",
                signature=signature,
                input_example=input_example
            )

            if rmse < best_score:
                best_score, best_name, best_model = rmse, name, model

    return best_name, best_model, {"rmse": best_score, "feature_count": X_train.shape[1]}

def main():
    parser = argparse.ArgumentParser(
        description="Train Iris classifier or California Housing regressor"
    )
    parser.add_argument(
        "--task", choices=["iris", "housing"],
        required=True, help="Choose 'iris' or 'housing'"
    )
    args = parser.parse_args()

    setup_mlflow()

    os.makedirs("models", exist_ok=True)

    if args.task == "iris":
        name, model, metrics = train_classification()
        model_path = "models/iris_best_model.pkl"
        scaler_path = "models/iris_scaler.pkl"
    else:
        name, model, metrics = train_regression()
        model_path = "models/housing_best_model.pkl"
        scaler_path = "models/housing_scaler.pkl"

    # Save best model and corresponding scaler
    joblib.dump(model, model_path)
    # The scalers are saved in data_loader during preprocessing as 'models/scaler.pkl'.
    # Copy/rename them to task-specific filenames here
    if os.path.exists("models/scaler.pkl"):
        os.rename("models/scaler.pkl", scaler_path)

    with open(f"models/{args.task}_metrics.json", "w") as f:
        metrics_to_save = {"task": args.task, "best_model": name, **metrics}
        json.dump(metrics_to_save, f, indent=2)

    print(f"[{args.task}] Best model: {name} | Metrics: {metrics}")

if __name__ == "__main__":
    main()
