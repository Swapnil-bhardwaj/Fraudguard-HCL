"""FastAPI REST application exposing health, metadata, and fraud scoring endpoints."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd

from src.persistence import fetch_audit_logs, load_model_pipeline, log_audit_entry
from src.scoring import score_single_record, score_batch_dataframe, DISCLAIMER_TEXT
from src.train_models import ModelPipeline

app = FastAPI(
    title="FraudGuard AI — Transaction Anomaly Detection Service",
    description="Enterprise REST API for high-throughput transaction anomaly scoring, risk triage, and compliance audit trail.",
    version="2.4.0",
)

MODELS_CACHE: Dict[str, ModelPipeline] = {}
MODELS_DIR = Path("models/v1")

MODEL_SLUG_MAP = {
    "isolation_forest": ("Isolation Forest", "isolation_forest.joblib"),
    "local_outlier_factor": ("Local Outlier Factor", "local_outlier_factor.joblib"),
    "autoencoder": ("Autoencoder", "autoencoder.joblib"),
    "logistic_regression_(supervised)": ("Logistic Regression (Supervised)", "logistic_regression_(supervised).joblib"),
    "logistic_regression_(smote)": ("Logistic Regression (SMOTE)", "logistic_regression_(smote).joblib"),
}


def get_loaded_pipeline(model_name: Optional[str] = None) -> ModelPipeline:
    """Retrieve or lazily load requested model pipeline."""
    target_name = (model_name or "Isolation Forest").strip().lower()

    # Match by key or display name
    matched_file = None
    for slug, (display_name, fname) in MODEL_SLUG_MAP.items():
        if target_name in [slug, display_name.lower(), fname.replace(".joblib", "").lower()]:
            matched_file = fname
            break
        if slug.startswith(target_name) or display_name.lower().startswith(target_name):
            matched_file = fname
            break

    if not matched_file:
        matched_file = "isolation_forest.joblib"

    if matched_file in MODELS_CACHE:
        return MODELS_CACHE[matched_file]

    model_path = MODELS_DIR / matched_file
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact '{matched_file}' not found. Run pipeline training first."
        )

    pipeline = load_model_pipeline(model_path)
    MODELS_CACHE[matched_file] = pipeline
    return pipeline


class TransactionInput(BaseModel):
    trans_id: Optional[int] = Field(default=None, description="Sequential transaction identifier")
    timestamp: str = Field(..., description="Timestamp in YYYY-MM-DD HH:MM:SS format", example="2024-01-05 14:30:00")
    card_token: str = Field(..., description="Masked card token CARD_XXXX####", example="CARD_XXXX1234")
    amount: float = Field(..., ge=0.0, description="Transaction amount (non-negative)", example=145.50)
    merchant_category: str = Field(..., description="Category (Healthcare, Fuel, Travel, Grocery, Online, Electronics, Restaurant)", example="Online")
    merchant_id: str = Field(..., description="Merchant identifier (e.g. MER0045)", example="MER0045")
    card_type: str = Field(..., description="Card type (Credit, Debit, Prepaid)", example="Credit")
    model_name: Optional[str] = Field(default="Isolation Forest", description="Model to use for scoring")


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]
    model_name: Optional[str] = Field(default="Isolation Forest")


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Return API health status, loaded models, and disclaimer."""
    persisted_models = [f.stem for f in MODELS_DIR.glob("*.joblib")] if MODELS_DIR.exists() else []
    return {
        "status": "healthy",
        "service": "FraudGuard AI Enterprise Transaction Anomaly Detection REST Service",
        "version": "2.4.0",
        "persisted_models_available": persisted_models,
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.get("/models", tags=["Models"])
def list_models() -> Dict[str, Any]:
    """List available model architectures and their operational thresholds."""
    available = []
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.joblib"):
            try:
                p = get_loaded_pipeline(f.stem)
                available.append({
                    "model_name": p.model_name,
                    "threshold": round(float(p.threshold), 5),
                    "score_orientation": p.score_orientation,
                    "features_count": len(p.preprocessor.output_feature_names_),
                })
            except Exception as e:
                available.append({"file": f.name, "error": str(e)})
    return {"models": available}


@app.post("/predict", tags=["Inference"])
def predict_single(item: TransactionInput) -> Dict[str, Any]:
    """Score a single credit card transaction and return human triage risk label."""
    t0 = time.perf_counter()
    pipeline = get_loaded_pipeline(item.model_name)
    record = item.model_dump()
    result = score_single_record(record, pipeline)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if not result.get("is_valid", False):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"errors": result.get("errors", ["Validation failed"])}
        )

    # Log to SQLite audit database
    log_audit_entry(
        trans_id=result.get("trans_id"),
        model_name=pipeline.model_name,
        anomaly_score=result["anomaly_score"],
        threshold=result["threshold"],
        flag=result["flag"],
        risk_label=result["risk_label"],
        execution_time_ms=latency_ms,
    )
    result["latency_ms"] = round(latency_ms, 3)
    return result


@app.post("/predict-batch", tags=["Inference"])
def predict_batch(batch: BatchTransactionInput) -> Dict[str, Any]:
    """Score a batch of transactions and return summary metrics and flagged items."""
    t0 = time.perf_counter()
    pipeline = get_loaded_pipeline(batch.model_name)
    records = [t.model_dump() for t in batch.transactions]
    if not records:
        raise HTTPException(status_code=400, detail="Batch transactions list is empty.")

    df_in = pd.DataFrame(records)
    try:
        scored_df = score_batch_dataframe(df_in, pipeline)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Batch scoring failed: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    total = len(scored_df)
    flagged_count = int(scored_df["flag"].sum())

    # Log each into audit table
    for _, row in scored_df.iterrows():
        log_audit_entry(
            trans_id=row.get("trans_id"),
            model_name=pipeline.model_name,
            anomaly_score=float(row["anomaly_score"]),
            threshold=float(row["threshold"]),
            flag=int(row["flag"]),
            risk_label=str(row["risk_label"]),
            execution_time_ms=latency_ms / total,
        )

    scored_records = scored_df[[
        "trans_id", "timestamp", "card_token", "amount", "merchant_category",
        "anomaly_score", "threshold", "flag", "risk_label", "neutral_indicator", "human_review_required"
    ]].to_dict(orient="records")

    return {
        "model_name": pipeline.model_name,
        "total_records": total,
        "flagged_for_review": flagged_count,
        "flagged_rate": round(flagged_count / total, 4) if total > 0 else 0.0,
        "batch_latency_ms": round(latency_ms, 3),
        "results": scored_records,
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.get("/audit-logs", tags=["Audit"])
def read_audit_logs(limit: int = 50) -> Dict[str, Any]:
    """Fetch recent audit log entries (strictly anonymized, zero PII)."""
    df = fetch_audit_logs(limit=limit)
    return {"count": len(df), "logs": df.to_dict(orient="records")}
