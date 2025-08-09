import argparse
import json
import os
import shutil

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import (load_and_preprocess_housing,
                             load_and_preprocess_iris)
from src.mlflow_config import setup_mlflow


def train_classification():
    X_train, X_test, y_train, y_test = load_and_preprocess_iris()
    models = {
        "logistic_regression": LogisticRegression(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    best_acc, best_name, best_model = 0, None, None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted")

            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=f"iris_{name}",
                signature=signature,
                input_example=X_test[:5],
            )

            if acc > best_acc:
                best_acc, best_name, best_model = acc, name, model

    return (
        best_name,
        best_model,
        {"accuracy": best_acc, "feature_count": X_train.shape[1]},
    )


def train_regression():
    X_train, X_test, y_train, y_test = load_and_preprocess_housing()
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    best_rmse, best_name, best_model = float("inf"), None, None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)

            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.log_params(model.get_params())
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=f"housing_{name}",
                signature=signature,
                input_example=X_test[:5],
            )

            if rmse < best_rmse:
                best_rmse, best_name, best_model = rmse, name, model

    return best_name, best_model, {"rmse": best_rmse, "feature_count": X_train.shape[1]}


def main():
    parser = argparse.ArgumentParser(description="Train ML models for Iris or Housing")
    parser.add_argument(
        "--task",
        choices=["iris", "housing"],
        required=True,
        help="Choose 'iris' or 'housing'",
    )
    args = parser.parse_args()

    setup_mlflow()
    os.makedirs("models", exist_ok=True)

    if args.task == "iris":
        name, model, metrics = train_classification()
        model_path = "models/iris_best_model.pkl"
        scaler_src = "models/iris_scaler.pkl"  # data_loader should save this
        metrics_file = "metrics_iris.json"
    else:
        name, model, metrics = train_regression()
        model_path = "models/housing_best_model.pkl"
        scaler_src = "models/housing_scaler.pkl"
        metrics_file = "metrics_housing.json"

    # Save best model
    joblib.dump(model, model_path)

    # Ensure scaler exists for this task
    default_scaler = "models/scaler.pkl"
    if os.path.exists(default_scaler) and not os.path.exists(scaler_src):
        shutil.copy(
            default_scaler, scaler_src
        )  # copy so it's still available for other tasks

    # Save metrics JSON using correct name for DVC
    with open(metrics_file, "w") as f:
        json.dump({"task": args.task, "best_model": name, **metrics}, f, indent=2)

    print(f"[{args.task}] Best model: {name} | Metrics: {metrics}")


if __name__ == "__main__":
    main()
