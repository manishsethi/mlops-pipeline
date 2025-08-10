import pytest
import joblib
import numpy as np
from src.Fast_api import MODELS, SCALERS

@pytest.fixture(scope="module")
def iris_model():
    return MODELS.get("iris")

@pytest.fixture(scope="module")
def iris_scaler():
    return SCALERS.get("iris")

def test_iris_model_loaded(iris_model):
    assert iris_model is not None, "Iris model should be loaded"

def test_iris_scaler_loaded(iris_scaler):
    assert iris_scaler is not None, "Iris scaler should be loaded"

def test_iris_prediction(iris_model, iris_scaler):
    # Example features for iris
    features = np.array([[5.1, 3.5, 1.4, 0.2]])
    scaled = iris_scaler.transform(features)
    prediction = iris_model.predict(scaled)
    assert prediction is not None
    assert prediction.shape[0] == 1

def test_iris_prediction_probabilities(iris_model, iris_scaler):
    if hasattr(iris_model, "predict_proba"):
        features = np.array([[5.1, 3.5, 1.4, 0.2]])
        scaled = iris_scaler.transform(features)
        probas = iris_model.predict_proba(scaled)
        assert probas is not None
        assert probas.shape[0] == 1
