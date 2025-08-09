# src/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import sqlite3

PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('ml_errors_total', 'Total number of errors', ['error_type'])
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')


class MetricsCollector:
    def __init__(self, db_path='predictions.db'):
        self.db_path = db_path

    def record_prediction(self, latency):
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(latency)

    def record_error(self, error_type):
        ERROR_COUNTER.labels(error_type=error_type).inc()

    def update_model_accuracy(self, accuracy):
        MODEL_ACCURACY.set(accuracy)

    def get_metrics_summary(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(response_time) as avg_response_time,
                    MIN(response_time) as min_response_time,
                    MAX(response_time) as max_response_time
                FROM predictions 
                WHERE datetime(timestamp) > datetime('now', '-24 hours')
                '''
            )
            result = cursor.fetchone()
            conn.close()

            return {
                'total_predictions_24h': result[0] if result and result[0] else 0,
                'avg_response_time': result[1] if result and result[1] else 0,
                'min_response_time': result[2] if result and result[2] else 0,
                'max_response_time': result[3] if result and result[3] else 0
            }
        except Exception as e:
            return {'error': str(e)}


metrics_collector = MetricsCollector()


# ========= FASTAPI COMPATIBLE METRICS ENDPOINT =========
from fastapi.responses import Response as FastAPIResponse

def metrics_endpoint_fastapi():
    """Prometheus metrics endpoint for FastAPI"""
    return FastAPIResponse(content=generate_latest(),
                           media_type=CONTENT_TYPE_LATEST)