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
import pytest
import joblib
import numpy as np
from sklearn.datasets import load_iris
from src.train import train_classification, train_regression

def test_train_classification():
    """Test training an Iris model"""
    name, model, metrics = train_classification()
    assert model is not None
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0  # simple sanity check

def test_train_regression():
    """Test training a Housing model"""
    name, model, metrics = train_regression()
    assert model is not None
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0  # simple sanity check

def test_model_prediction_iris():
    """Test prediction with saved Iris model"""
    try:
        model = joblib.load("models/iris_best_model.pkl")
        scaler = joblib.load("models/iris_scaler.pkl")
    except FileNotFoundError:
        pytest.skip("Iris model or scaler file not found")

    iris = load_iris()
    sample = iris.data[0:1]
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]  # valid Iris classes

def test_model_prediction_housing():
    """Test prediction with saved Housing model"""
    try:
        model = joblib.load("models/housing_best_model.pkl")
        scaler = joblib.load("models/housing_scaler.pkl")
    except FileNotFoundError:
        pytest.skip("Housing model or scaler file not found")

    # California housing has 8 features
    sample = np.random.rand(1, 8)
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)

    assert len(prediction) == 1
