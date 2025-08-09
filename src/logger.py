# src/logger.py
import logging
import sqlite3
from datetime import datetime
import json
import os


class SQLiteHandler(logging.Handler):
    """Custom logging handler for SQLite"""

    def __init__(self, db_path="logs/app_logs.db"):
        super().__init__()
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initialize SQLite database for logs"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                level TEXT,
                message TEXT,
                module TEXT,
                function TEXT,
                line_number INTEGER
            )
        """
        )
        conn.commit()
        conn.close()

    def emit(self, record):
        """Emit a log record to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO logs (timestamp, level, message, module, function, line_number)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.fromtimestamp(record.created).isoformat(),
                    record.levelname,
                    self.format(record),
                    record.module,
                    record.funcName,
                    record.lineno,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            self.handleError(record)


def setup_logging():
    """Setup comprehensive logging configuration"""
    # Create logger
    logger = logging.getLogger("ml_api")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # File handler
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # SQLite handler
    sqlite_handler = SQLiteHandler()
    sqlite_handler.setLevel(logging.INFO)
    sqlite_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(sqlite_handler)

    return logger
