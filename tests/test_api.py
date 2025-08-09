# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.Fast_api import app  # Import your FastAPI app
import os, joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from src.Fast_api import app, MODELS, SCALERS

client = TestClient(app)  # Instantiate TestClient for your FastAPI app

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert data['status'] == 'healthy'

@pytest.fixture(autouse=True)
def prepare_artifacts(tmp_path):
    # Create a trivial model and scaler for Iris
    model = DummyClassifier(strategy="most_frequent")
    scaler = lambda X: X  # identity

    # Save to models folder
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_best_model.pkl")
    joblib.dump(scaler, "models/iris_scaler.pkl")

    # Force reload artifacts in the running app
    MODELS["iris"] = model
    SCALERS["iris"] = scaler

    yield

def test_predict_endpoint_valid_input():
    test_data = {
        "task": "iris",
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'response_time_seconds' in data

def test_predict_endpoint_invalid_input():
    test_data = {
        "task": "iris",
        "features": [1, 2]  # Wrong number of features
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # FastAPI validation error status code

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # /metrics returns raw Prometheus text format, so response.json() will fail,
    # so just check content type
    assert "text/plain" in response.headers.get("content-type", "")

