# src/mlflow_config.py
import os

import mlflow
import mlflow.sklearn


def setup_mlflow():
    """Configure MLflow tracking"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ml-pipeline-experiment")

    return mlflow
