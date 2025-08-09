# src/retraining.py
import logging
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd

from src.train import train_classification, train_regression


class ModelRetrainingSystem:
    """Automated model retraining system"""

    def __init__(
        self,
        performance_threshold=0.85,
        drift_threshold=0.1,
        min_samples=100,
        model_type="classification",
    ):
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)

    def check_performance_drift(self):
        """Check for model performance drift"""
        try:
            conn = sqlite3.connect("predictions.db")
            # Get recent predictions (last 7 days)
            query = """
                SELECT input_data, prediction, timestamp
                FROM predictions 
                WHERE datetime(timestamp) > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if len(df) < self.min_samples:
                self.logger.info(f"Insufficient samples for drift detection: {len(df)}")
                return False, "insufficient_data"

            # Calculate performance metrics (this is a simplified example)
            # In practice, you'd need ground truth labels for proper evaluation
            recent_performance = self._calculate_recent_performance(df)

            if recent_performance < self.performance_threshold:
                self.logger.warning(
                    f"Performance drift detected: {recent_performance:.3f} < {self.performance_threshold}"
                )
                return True, "performance_drift"

            return False, "no_drift"

        except Exception as e:
            self.logger.error(f"Error checking performance drift: {str(e)}")
            return False, "error"

    def _calculate_recent_performance(self, df):
        """Calculate recent model performance (simplified)"""
        # This is a placeholder - you'd need actual ground truth labels
        # For demo purposes, we'll use response time as a proxy
        avg_response_time = (
            df["response_time"].mean() if "response_time" in df.columns else 0.1
        )

        # Simulate performance metric (higher response time = lower performance)
        simulated_performance = max(0, 1.0 - (avg_response_time - 0.05) * 10)
        return simulated_performance

    def trigger_retraining(self):
        """Trigger model retraining process"""
        try:
            self.logger.info("Starting automated model retraining...")

            self._backup_current_model()

            # Retrain model based on type
            if self.model_type == "classification":
                _, new_model, _ = train_classification()
            else:
                _, new_model, _ = train_regression()

            if new_model is not None:
                self.logger.info("Model retraining completed successfully")

                if self._validate_new_model(new_model):
                    self.logger.info("New model validated and deployed")
                    return True
                else:
                    self.logger.warning("New model validation failed, rolling back")
                    self._rollback_model()
                    return False
            else:
                self.logger.error("Model retraining failed")
                return False

        except Exception as e:
            self.logger.error(f"Retraining process failed: {str(e)}")
            return False

    def _backup_current_model(self):
        """Backup current model before retraining"""
        try:
            import shutil

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy("models/best_model.pkl", f"models/backup_model_{timestamp}.pkl")
            shutil.copy("models/scaler.pkl", f"models/backup_scaler_{timestamp}.pkl")
            self.logger.info(f"Model backed up with timestamp: {timestamp}")
        except Exception as e:
            self.logger.error(f"Failed to backup model: {str(e)}")

    def _validate_new_model(self, model):
        """Validate newly trained model"""
        try:
            # Perform basic validation tests
            # This is simplified - you'd want more comprehensive validation

            # Test prediction capability
            if self.model_type == "classification":
                test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
            else:
                test_input = np.array(
                    [[8.3252, 41.0, 6.98, 1.02, 322.0, 2.56, 37.88, -122.23]]
                )

            scaler = joblib.load("models/scaler.pkl")
            test_scaled = scaler.transform(test_input)
            prediction = model.predict(test_scaled)

            return prediction is not None and len(prediction) > 0

        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return False

    def _rollback_model(self):
        """Rollback to previous model version"""
        try:
            import glob
            import shutil

            # Find most recent backup
            backup_files = glob.glob("models/backup_model_*.pkl")
            if backup_files:
                latest_backup = max(backup_files)
                shutil.copy(latest_backup, "models/best_model.pkl")

                # Also restore scaler
                scaler_backup = latest_backup.replace("backup_model_", "backup_scaler_")
                if os.path.exists(scaler_backup):
                    shutil.copy(scaler_backup, "models/scaler.pkl")

                self.logger.info("Model rolled back successfully")
            else:
                self.logger.error("No backup found for rollback")

        except Exception as e:
            self.logger.error(f"Model rollback failed: {str(e)}")


# Retraining scheduler
def schedule_drift_check():
    """Schedule periodic drift checking"""
    retraining_system = ModelRetrainingSystem()

    needs_retraining, reason = retraining_system.check_performance_drift()

    if needs_retraining:
        logging.info(f"Retraining triggered due to: {reason}")
        success = retraining_system.trigger_retraining()

        if success:
            logging.info("Automated retraining completed successfully")
        else:
            logging.error("Automated retraining failed")
    else:
        logging.info(f"No retraining needed: {reason}")
