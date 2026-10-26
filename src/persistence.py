"""Model artifact persistence, metadata serialization, and SQLite audit logging."""
from __future__ import annotations
import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import joblib
import pandas as pd

from src.train_models import ModelPipeline
from src.utils import NumpyJSONEncoder, setup_logger

logger = setup_logger("persistence")


def get_model_slug(name: str) -> str:
    """Convert human-readable model name to file-safe slug."""
    return name.lower().replace(" ", "_").replace("/", "_")


def save_model_pipeline(
    pipeline: ModelPipeline,
    save_dir: Path = Path("models"),
    version: str = "v1"
) -> Path:
    """Persist fitted ModelPipeline artifact and JSON metadata."""
    target_dir = save_dir / version
    target_dir.mkdir(parents=True, exist_ok=True)

    slug = get_model_slug(pipeline.model_name)
    model_file = target_dir / f"{slug}.joblib"
    meta_file = target_dir / f"{slug}_metadata.json"

    # Save fitted pipeline bundle
    joblib.dump(pipeline, model_file, compress=3)

    # Save metadata JSON
    meta = {
        "model_name": pipeline.model_name,
        "slug": slug,
        "version": version,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "threshold": float(pipeline.threshold),
        "score_orientation": pipeline.score_orientation,
        "feature_count": len(pipeline.preprocessor.output_feature_names_),
        "features": pipeline.preprocessor.output_feature_names_,
        "extra_metadata": pipeline.metadata,
    }
    meta_file.write_text(json.dumps(meta, indent=2, cls=NumpyJSONEncoder), encoding="utf-8")
    logger.info(f"Saved model pipeline to {model_file}")
    return model_file


def load_model_pipeline(model_path: Path) -> ModelPipeline:
    """Load a persisted ModelPipeline artifact."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline: ModelPipeline = joblib.load(model_path)
    logger.info(f"Loaded model pipeline '{pipeline.model_name}' from {model_path}")
    return pipeline


def init_audit_db(db_path: Path = Path("outputs/audit.db")) -> None:
    """
    Initialize SQLite audit table.
    STRICT PRIVACY POLICY: Never stores card numbers, CVVs, expiry dates, or customer names.
    Stores only minimal metadata, operational telemetry, and anomaly scores.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            trans_id INTEGER,
            model_name TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            threshold REAL NOT NULL,
            flag INTEGER NOT NULL,
            risk_label TEXT NOT NULL,
            execution_time_ms REAL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_audit_entry(
    trans_id: Optional[int],
    model_name: str,
    anomaly_score: float,
    threshold: float,
    flag: int,
    risk_label: str,
    execution_time_ms: float = 0.0,
    db_path: Path = Path("outputs/audit.db"),
) -> None:
    """Insert an anonymized operational log record into SQLite audit log."""
    try:
        init_audit_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor.execute(
            """
            INSERT INTO audit_logs (
                timestamp, trans_id, model_name, anomaly_score, threshold, flag, risk_label, execution_time_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                int(trans_id) if trans_id is not None else None,
                model_name,
                float(anomaly_score),
                float(threshold),
                int(flag),
                risk_label,
                float(execution_time_ms),
                now_str,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to record audit entry: {e}")


def fetch_audit_logs(limit: int = 50, db_path: Path = Path("outputs/audit.db")) -> pd.DataFrame:
    """Retrieve recent anonymized audit records."""
    if not db_path.exists():
        return pd.DataFrame(columns=[
            "id", "timestamp", "trans_id", "model_name", "anomaly_score",
            "threshold", "flag", "risk_label", "execution_time_ms", "created_at"
        ])
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?",
        conn,
        params=(limit,)
    )
    conn.close()
    return df
