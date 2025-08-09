# src/mlflow_config.py
import mlflow
import mlflow.sklearn
import os


def setup_mlflow():
    """Configure MLflow tracking"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ml-pipeline-experiment")

    return mlflow
