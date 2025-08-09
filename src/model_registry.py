# src/model_registry.py
import mlflow
from mlflow.tracking import MlflowClient

def register_best_model(model_name, run_id):
    """Register the best model in MLflow Model Registry"""
    client = MlflowClient()
    
    # Register model
    model_version = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )
    
    # Transition to production
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    return model_version
