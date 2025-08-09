# tests/test_api.py
import pytest
import json
import numpy as np
from src.api import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_predict_endpoint_valid_input(client):
    """Test prediction with valid input"""
    test_data = {
        'features': [5.1, 3.5, 1.4, 0.2]  # Sample iris features
    }
    
    response = client.post('/predict', 
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'response_time_seconds' in data

def test_predict_endpoint_invalid_input(client):
    """Test prediction with invalid input"""
    test_data = {
        'features': [1, 2]  # Wrong number of features
    }
    
    response = client.post('/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 400

def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get('/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'total_predictions' in data
