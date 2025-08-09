# src/Fast_api.py
import logging
import os
import joblib
import numpy as np
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationInfo, field_validator

from src.metrics import metrics_endpoint_fastapi

# =========================================================
# App Initialization
# =========================================================
app = FastAPI(
    title="ML Model API",
    version="2.0.0",
    description="API for serving multiple ML models (Iris classification & Housing regression).",
)

# =========================================================
# Configuration
# =========================================================
# Map of task -> expected feature count
EXPECTED_FEATURES = {"iris": 4, "housing": 8}

# Model and scaler storage
MODELS = {}
SCALERS = {}


# =========================================================
# Load Models & Scalers at Startup
# =========================================================
def load_artifacts():
    for task in EXPECTED_FEATURES.keys():
        model_path = f"models/{task}_best_model.pkl"
        scaler_path = f"models/{task}_scaler.pkl"

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                MODELS[task] = joblib.load(model_path)
                SCALERS[task] = joblib.load(scaler_path)
                logging.info(f"Loaded {task} model and scaler.")
            except Exception as e:
                logging.error(f"Error loading {task} artifacts: {e}")
        else:
            logging.warning(
                f"Artifacts for task '{task}' not found: "
                f"{model_path}, {scaler_path}"
            )


load_artifacts()


# =========================================================
# Request & Response Schemas
# =========================================================
class PredictionRequest(BaseModel):
    task: str
    features: List[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v, info: ValidationInfo):
        task = info.data.get("task")  # access other fields from info.data dictionary
        if task not in EXPECTED_FEATURES:
            raise ValueError(
                f"Unknown task '{task}'. Must be one of {list(EXPECTED_FEATURES.keys())}"
            )
        if len(v) != EXPECTED_FEATURES[task]:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES[task]} features for '{task}', got {len(v)}"
            )
        return v


class PredictionResponse(BaseModel):
    task: str
    prediction: List[float]
    prediction_probabilities: Optional[List[List[float]]] = None
    response_time_seconds: float
    timestamp: str


# =========================================================
# Routes
# =========================================================
@app.get("/")
async def root():
    return {
        "message": "Welcome to the ML Model API",
        "available_endpoints": [
            "/predict (POST)",
            "/health (GET)",
            "/docs",
            "/metrics (GET)",
        ],
    }


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return metrics_endpoint_fastapi()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "loaded_models": list(MODELS.keys()),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = datetime.now()

    # Get matching model and scaler
    model = MODELS.get(request.task)
    scaler = SCALERS.get(request.task)

    if model is None or scaler is None:
        raise HTTPException(
            status_code=500, detail=f"No model/scaler loaded for task '{request.task}'."
        )

    try:
        # Prepare input
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        prediction_proba = None
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(features_scaled).tolist()

        response_time = (datetime.now() - start_time).total_seconds()

        return PredictionResponse(
            task=request.task,
            prediction=prediction.tolist(),
            prediction_probabilities=prediction_proba,
            response_time_seconds=response_time,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
