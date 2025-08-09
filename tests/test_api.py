# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.Fastapi_app import app  # Import your FastAPI app

client = TestClient(app)  # Instantiate TestClient for your FastAPI app

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert data['status'] == 'healthy'

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
