# src/Fast_api.py
import os
import joblib
import numpy as np
import logging
import sqlite3
import json
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationInfo, field_validator

from src.metrics import metrics_collector  # your existing metrics collector object
from src.metrics import metrics_endpoint_fastapi

# =========================================================
# Logging Configuration
# =========================================================
# Setup logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),  # Log file
        logging.StreamHandler()  # Console output
    ],
)

logger = logging.getLogger("ml_model_api")

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


def log_prediction(task: str, features: list, prediction: list, response_time: float):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT,
            input_data TEXT,
            prediction TEXT,
            response_time REAL,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO predictions (task, input_data, prediction, response_time, timestamp)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (task, json.dumps(features), json.dumps(prediction), response_time))
    conn.commit()
    conn.close()


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

    model = MODELS.get(request.task)
    scaler = SCALERS.get(request.task)

    if model is None or scaler is None:
        metrics_collector.record_error(error_type="model_missing")
        raise HTTPException(status_code=500, detail=f"No model/scaler loaded for task '{request.task}'.")

    try:
        features = np.array(request.features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled).tolist() if hasattr(model, "predict_proba") else None

        response_time = (datetime.now() - start_time).total_seconds()

        # === Log prediction persistently to SQLite ===
        log_prediction(request.task, request.features, prediction.tolist(), response_time)

        # === Record Prometheus metrics ===
        metrics_collector.record_prediction(latency=response_time)

        return PredictionResponse(
            task=request.task,
            prediction=prediction.tolist(),
            prediction_probabilities=prediction_proba,
            response_time_seconds=response_time,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        metrics_collector.record_error(error_type="prediction_failure")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
