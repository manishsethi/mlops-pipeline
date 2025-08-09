# tests/test_model.py
import pytest
import joblib
import numpy as np
from sklearn.datasets import load_iris
from src.train import train_models

def test_model_training():
    """Test model training process"""
    # This would be a more comprehensive test in practice
    model = train_models()
    assert model is not None
    
def test_model_prediction():
    """Test model prediction"""
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Test with sample data
        iris = load_iris()
        sample = iris.data[0:1]
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)
        
        assert len(prediction) == 1
        assert prediction[0] in [0, 1, 2]  # Valid iris classes
        
    except FileNotFoundError:
        pytest.skip("Model files not found")
