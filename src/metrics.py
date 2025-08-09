# src/metrics.py
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
import sqlite3

# ===== PROMETHEUS METRICS DEFINITIONS =====

PREDICTION_COUNTER = Counter(
    "ml_predictions_total", "Total number of predictions made"
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_duration_seconds", "Prediction latency in seconds"
)
ERROR_COUNTER = Counter(
    "ml_errors_total", "Total number of errors encountered", ["error_type"]
)
MODEL_ACCURACY = Gauge(
    "ml_model_accuracy", "Current deployed model's accuracy"
)

# Gauges for DB-backed metrics
DB_TOTAL_PREDICTIONS_24H = Gauge(
    "ml_total_predictions_24h", "Total predictions served in the last 24 hours"
)
DB_AVG_RESPONSE_TIME_24H = Gauge(
    "ml_avg_response_time_24h", "Average prediction response time in the last 24 hours"
)
DB_MIN_RESPONSE_TIME_24H = Gauge(
    "ml_min_response_time_24h", "Minimum prediction response time in the last 24 hours"
)
DB_MAX_RESPONSE_TIME_24H = Gauge(
    "ml_max_response_time_24h", "Maximum prediction response time in the last 24 hours"
)


# ===== METRICS COLLECTOR CLASS =====

class MetricsCollector:
    def __init__(self, db_path="predictions.db"):
        self.db_path = db_path

    def record_prediction(self, latency: float):
        """Record a prediction event and its latency."""
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(latency)

    def record_error(self, error_type: str):
        """Record an error event."""
        ERROR_COUNTER.labels(error_type=error_type).inc()

    def update_model_accuracy(self, accuracy: float):
        """Update the model accuracy gauge."""
        MODEL_ACCURACY.set(accuracy)

    def get_metrics_summary(self):
        """Query SQLite DB for last 24h prediction stats."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task TEXT,
                    input_data TEXT,
                    prediction TEXT,
                    response_time REAL,
                    timestamp TEXT
                )
                """
            )
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(response_time) as avg_response_time,
                    MIN(response_time) as min_response_time,
                    MAX(response_time) as max_response_time
                FROM predictions 
                WHERE datetime(timestamp) > datetime('now', '-24 hours')
                """
            )
            result = cursor.fetchone()
            conn.close()

            summary = {
                "total_predictions_24h": result[0] if result and result[0] else 0,
                "avg_response_time": result[1] if result and result[1] else 0,
                "min_response_time": result[2] if result and result[2] else 0,
                "max_response_time": result[3] if result and result[3] else 0,
            }

            # Push DB stats to Prometheus gauges
            DB_TOTAL_PREDICTIONS_24H.set(summary["total_predictions_24h"])
            DB_AVG_RESPONSE_TIME_24H.set(summary["avg_response_time"])
            DB_MIN_RESPONSE_TIME_24H.set(summary["min_response_time"])
            DB_MAX_RESPONSE_TIME_24H.set(summary["max_response_time"])

            return summary

        except Exception as e:
            return {"error": str(e)}


# Global instance used by the API
metrics_collector = MetricsCollector()


# ===== FASTAPI METRICS ENDPOINT =====
from fastapi.responses import Response as FastAPIResponse

def metrics_endpoint_fastapi():
    """
    Prometheus metrics endpoint for FastAPI.
    This will also refresh DB-based gauges each time it is called.
    """
    metrics_collector.get_metrics_summary()  # update gauges from DB
    return FastAPIResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
