# tests/test_fastapi.py

import os
import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.Fast_api import app, MODELS, SCALERS

# Fixture to create TestClient
@pytest.fixture(scope="module")
def client():
    return TestClient(app)

# Fixture to prepare minimal dummy artifacts for testing /predict endpoint
@pytest.fixture(scope="module", autouse=True)
def prepare_artifacts():
    # Create dummy model and scaler for 'iris' task
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import StandardScaler

    os.makedirs("models", exist_ok=True)

    iris_model = DummyClassifier(strategy="most_frequent")
    iris_model.fit([[0,0,0,0]], [0])  # fit dummy data
    iris_scaler = StandardScaler()
    iris_scaler.fit([[0,0,0,0]])  # fit dummy scaler

    joblib.dump(iris_model, "models/iris_best_model.pkl")
    joblib.dump(iris_scaler, "models/iris_scaler.pkl")

    MODELS["iris"] = iris_model
    SCALERS["iris"] = iris_scaler

    # Similarly for housing
    from sklearn.dummy import DummyRegressor

    housing_model = DummyRegressor()
    housing_model.fit([[0]*8], [0])
    housing_scaler = StandardScaler()
    housing_scaler.fit([[0]*8])

    joblib.dump(housing_model, "models/housing_best_model.pkl")
    joblib.dump(housing_scaler, "models/housing_scaler.pkl")

    MODELS["housing"] = housing_model
    SCALERS["housing"] = housing_scaler

    yield

    # Optionally cleanup after tests
    # os.remove("models/iris_best_model.pkl")
    # os.remove("models/iris_scaler.pkl")
    # os.remove("models/housing_best_model.pkl")
    # os.remove("models/housing_scaler.pkl")

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    json_resp = response.json()
    assert "message" in json_resp
    assert "available_endpoints" in json_resp

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "healthy"
    assert isinstance(json_resp.get("loaded_models"), list)
    assert "timestamp" in json_resp

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    # /metrics returns Prometheus plain text, check content type
    assert "text/plain" in response.headers.get("content-type", "")

def test_predict_valid_iris(client):
    test_payload = {
        "task": "iris",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["task"] == "iris"
    assert "prediction" in json_resp
    assert "response_time_seconds" in json_resp
    assert "timestamp" in json_resp

def test_predict_valid_housing(client):
    test_payload = {
        "task": "housing",
        "features": [0.5, 3.0, 1.5, 5.3, 2.2, 0.1, 8.7, 4.4]
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["task"] == "housing"
    assert "prediction" in json_resp

def test_predict_invalid_task(client):
    test_payload = {
        "task": "unknown_task",
        "features": [1,2,3,4]
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 422  # validation error due to invalid task

def test_predict_invalid_features_length(client):
    # Task is iris (expects 4 features), provide wrong number
    test_payload = {
        "task": "iris",
        "features": [1, 2]
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 422  # Pydantic validation error

def test_predict_model_or_scaler_not_loaded(client):
    # Temporarily remove model/scaler for a task to simulate error
    backup_model = MODELS.pop("iris", None)
    backup_scaler = SCALERS.pop("iris", None)

    test_payload = {
        "task": "iris",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=test_payload)
    assert response.status_code == 500
    assert "No model/scaler loaded" in response.json()["detail"]

    # Restore artifacts
    if backup_model:
        MODELS["iris"] = backup_model
    if backup_scaler:
        SCALERS["iris"] = backup_scaler
