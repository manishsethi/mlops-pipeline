# src/Fast_api.py

import os
import joblib
import numpy as np
import logging
import sqlite3
import json
import shutil
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationInfo, field_validator

from src.metrics import metrics_collector
from src.metrics import metrics_endpoint_fastapi

# =========================================================
# Logging Configuration
# =========================================================
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, "api.log")),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("ml_model_api")

# =========================================================
# Paths & Config
# =========================================================
# Determine a writable database path
if os.environ.get("TESTING") or not os.access("/app", os.W_OK):
    if os.environ.get("TESTING"):
        DB_PATH = ":memory:"
        logger.info("Using in-memory database for testing")
    else:
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)
        DB_PATH = os.path.join(data_dir, "predictions.db")
        logger.info(f"Using database at: {DB_PATH}")
else:
    DB_PATH = "/app/data/predictions.db"
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

EXPECTED_FEATURES = {"iris": 4, "housing": 8}

# If a directory exists at DB_PATH, remove it
if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
    logger.warning(f"{DB_PATH} exists as a directory — removing it")
    try:
        shutil.rmtree(DB_PATH)
    except Exception as e:
        logger.error(f"Failed to remove directory {DB_PATH}: {e}")

# =========================================================
# Initialize DB
# =========================================================
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
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
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {DB_PATH}")
    except sqlite3.OperationalError as e:
        logger.error(f"DB init failed: {e}. Falling back to in-memory DB.")
        global DB_PATH
        DB_PATH = ":memory:"
        conn = sqlite3.connect(DB_PATH)
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
        conn.commit()
        conn.close()

# Only initialize DB when not collecting tests
if not os.environ.get("PYTEST_CURRENT_TEST"):
    init_db()

# =========================================================
# FastAPI App Setup
# =========================================================
app = FastAPI(
    title="ML Model API",
    version="2.0.0",
    description="API for serving multiple ML models (Iris & Housing).",
)

MODELS = {}
SCALERS = {}

def load_artifacts():
    for task in EXPECTED_FEATURES:
        mpath = f"models/{task}_best_model.pkl"
        spath = f"models/{task}_scaler.pkl"
        if os.path.exists(mpath) and os.path.exists(spath):
            try:
                MODELS[task] = joblib.load(mpath)
                SCALERS[task] = joblib.load(spath)
                logger.info(f"Loaded artifacts for task '{task}'")
            except Exception as e:
                logger.error(f"Error loading artifacts for '{task}': {e}")
        else:
            logger.warning(f"Artifacts missing for '{task}': {mpath}, {spath}")

if not os.environ.get("PYTEST_CURRENT_TEST"):
    load_artifacts()

# =========================================================
# Prediction Logging Utility
# =========================================================
def log_prediction(task: str, features: list, prediction: list, response_time: float):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (task, input_data, prediction, response_time, timestamp)
            VALUES (?, ?, ?, ?, datetime('now'))
        """, (task, json.dumps(features), json.dumps(prediction), response_time))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

# =========================================================
# Pydantic Schemas
# =========================================================
class PredictionRequest(BaseModel):
    task: str
    features: List[float]

    @field_validator("features")
    @classmethod
    def validate_features(cls, v, info: ValidationInfo):
        task = info.data.get("task")
        if task not in EXPECTED_FEATURES:
            raise ValueError(f"Unknown task '{task}'. Choose from {list(EXPECTED_FEATURES.keys())}")
        if len(v) != EXPECTED_FEATURES[task]:
            raise ValueError(f"Expected {EXPECTED_FEATURES[task]} features for '{task}', got {len(v)}")
        return v

class PredictionResponse(BaseModel):
    task: str
    prediction: List[float]
    prediction_probabilities: Optional[List[List[float]]] = None
    response_time_seconds: float
    timestamp: str

# =========================================================
# API Endpoints
# =========================================================
@app.get("/")
async def root():
    return {
        "message": "Welcome to the ML Model API",
        "endpoints": ["/predict", "/health", "/metrics", "/docs"],
    }

@app.get("/metrics")
def metrics():
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
    start = datetime.now()
    model = MODELS.get(request.task)
    scaler = SCALERS.get(request.task)

    if model is None or scaler is None:
        metrics_collector.record_error(error_type="model_missing")
        raise HTTPException(500, f"No model/scaler for task '{request.task}'")

    try:
        arr = np.array(request.features).reshape(1, -1)
        scaled = scaler.transform(arr)

        preds = model.predict(scaled)
        probs = model.predict_proba(scaled).tolist() if hasattr(model, "predict_proba") else None

        rt = (datetime.now() - start).total_seconds()
        log_prediction(request.task, request.features, preds.tolist(), rt)
        metrics_collector.record_prediction(latency=rt)

        return PredictionResponse(
            task=request.task,
            prediction=preds.tolist(),
            prediction_probabilities=probs,
            response_time_seconds=rt,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        metrics_collector.record_error(error_type="prediction_failure")
        raise HTTPException(500, f"Prediction error: {e}")

# =========================================================
# Startup Event
# =========================================================
@app.on_event("startup")
async def on_startup():
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        init_db()
        load_artifacts()
        logger.info("Startup initialization complete")
