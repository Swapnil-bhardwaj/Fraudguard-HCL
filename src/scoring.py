"""Inference, anomaly scoring, risk-level assignment, and batch evaluation."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.data_validation import validate_single_transaction, validate_transaction_dataframe
from src.train_models import ModelPipeline

DISCLAIMER_TEXT = (
    "DISCLAIMER: This anomaly flag is a mathematical risk signal for human review and triage. "
    "It is NOT legal or factual proof of fraud, does not automatically decline a transaction, "
    "and must never be used to make unreviewed financial or credit decisions."
)


def derive_neutral_indicators(row: pd.Series, score: float, threshold: float) -> str:
    """
    Generate neutral, non-causal descriptive indicators explaining transaction context.
    Avoids claiming true causality while surfacing observable transaction anomalies.
    """
    signals: List[str] = []
    
    amount = float(row.get("amount", 0.0))
    if amount >= 1500.0:
        signals.append(f"Substantial transaction amount (${amount:,.2f})")
    elif amount >= 750.0:
        signals.append(f"Elevated amount (${amount:,.2f})")

    ts = pd.to_datetime(row.get("timestamp"))
    if not pd.isna(ts):
        hour = ts.hour
        if hour >= 23 or hour < 5:
            signals.append(f"Late-night transaction window ({hour:02d}:00)")

    category = str(row.get("merchant_category", ""))
    if category in ["Electronics", "Travel", "Online"]:
        signals.append(f"High-frequency dispute category ({category})")

    if not signals:
        if score >= threshold:
            return "Multi-dimensional feature vector deviation detected by model."
        return "Transaction parameters align with typical baseline distribution."

    return " | ".join(signals)


def score_single_record(record: Dict[str, Any], pipeline: ModelPipeline) -> Dict[str, Any]:
    """Score an individual transaction record using the loaded model pipeline."""
    is_valid, errors, validated_record = validate_single_transaction(record)
    if not is_valid:
        return {
            "is_valid": False,
            "errors": errors,
            "trans_id": record.get("trans_id"),
            "risk_label": "Validation Error",
            "flag": 0,
            "disclaimer": DISCLAIMER_TEXT,
        }

    df_single = pd.DataFrame([validated_record])
    raw_score = float(pipeline.score_raw_dataframe(df_single)[0])
    threshold = float(pipeline.threshold)
    is_flagged = int(raw_score >= threshold)
    risk_label = "Review recommended" if is_flagged == 1 else "Lower anomaly score"
    indicator = derive_neutral_indicators(pd.Series(validated_record), raw_score, threshold)

    return {
        "is_valid": True,
        "trans_id": validated_record.get("trans_id"),
        "timestamp": str(validated_record.get("timestamp")),
        "card_token": str(validated_record.get("card_token")),
        "amount": float(validated_record.get("amount")),
        "merchant_category": str(validated_record.get("merchant_category")),
        "merchant_id": str(validated_record.get("merchant_id")),
        "card_type": str(validated_record.get("card_type")),
        "model_name": pipeline.model_name,
        "anomaly_score": round(raw_score, 5),
        "threshold": round(threshold, 5),
        "flag": is_flagged,
        "risk_label": risk_label,
        "neutral_indicator": indicator,
        "human_triage_action": "Refer to Compliance Officer for manual verification" if is_flagged == 1 else "Standard automated clearing",
        "disclaimer": DISCLAIMER_TEXT,
    }


def score_batch_dataframe(df: pd.DataFrame, pipeline: ModelPipeline) -> pd.DataFrame:
    """Score a full DataFrame of transactions and append risk flags, scores, and indicators."""
    is_valid, errors, validated_df = validate_transaction_dataframe(
        df, require_target=False, allow_extra_columns=True
    )
    if not is_valid:
        raise ValueError(f"Batch data failed validation: {errors}")

    scores = pipeline.score_raw_dataframe(validated_df)
    threshold = float(pipeline.threshold)
    flags = (scores >= threshold).astype(int)

    out_df = validated_df.copy()
    out_df["anomaly_score"] = np.round(scores, 5)
    out_df["threshold"] = round(threshold, 5)
    out_df["model_used"] = pipeline.model_name
    out_df["flag"] = flags
    out_df["risk_label"] = np.where(flags == 1, "Review recommended", "Lower anomaly score")

    # Generate row-by-row neutral indicators
    indicators = [
        derive_neutral_indicators(row, score, threshold)
        for (_, row), score in zip(out_df.iterrows(), scores)
    ]
    out_df["neutral_indicator"] = indicators
    out_df["human_review_required"] = np.where(flags == 1, "YES", "NO")
    return out_df
