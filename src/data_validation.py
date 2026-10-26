"""Comprehensive data validation and quality reporting for transaction datasets."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "trans_id", "timestamp", "card_token", "amount",
    "merchant_category", "is_fraud", "merchant_id", "card_type"
]

INFERENCE_REQUIRED_COLUMNS = [
    "trans_id", "timestamp", "card_token", "amount",
    "merchant_category", "merchant_id", "card_type"
]

VALID_CATEGORIES: Set[str] = {
    "Healthcare", "Fuel", "Travel", "Grocery", "Online", "Electronics", "Restaurant"
}

VALID_CARD_TYPES: Set[str] = {
    "Credit", "Debit", "Prepaid"
}


class ValidationError(Exception):
    """Raised when dataset fails critical schema or validity rules."""
    pass


def validate_transaction_dataframe(
    df: pd.DataFrame,
    require_target: bool = True,
    allow_extra_columns: bool = False
) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validate a transaction DataFrame against structural and domain rules.

    Returns:
        (is_valid, errors, validated_df)
    """
    errors: List[str] = []
    if not isinstance(df, pd.DataFrame):
        return False, ["Input is not a valid pandas DataFrame."], pd.DataFrame()

    df_copy = df.copy()

    # Check required columns
    required = EXPECTED_COLUMNS if require_target else INFERENCE_REQUIRED_COLUMNS
    missing_cols = [c for c in required if c not in df_copy.columns]
    if missing_cols:
        errors.append(f"Missing required column(s): {missing_cols}")

    if not allow_extra_columns:
        extra_cols = [c for c in df_copy.columns if c not in (EXPECTED_COLUMNS if require_target else (INFERENCE_REQUIRED_COLUMNS + ["is_fraud"]))]
        if extra_cols:
            errors.append(f"Unexpected extra column(s) detected: {extra_cols}. Expected schema: {required}")

    if errors:
        return False, errors, df_copy

    # Check empty DataFrame
    if len(df_copy) == 0:
        return False, ["Dataset contains zero rows."], df_copy

    # Check missing values
    null_counts = df_copy[required].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        errors.append(f"Missing values (nulls) detected in columns: {cols_with_nulls}")

    # Check trans_id
    try:
        df_copy["trans_id"] = pd.to_numeric(df_copy["trans_id"], errors="raise").astype("int64")
        if df_copy["trans_id"].duplicated().any():
            dup_count = int(df_copy["trans_id"].duplicated().sum())
            errors.append(f"Duplicate trans_id found ({dup_count} duplicates). trans_id must be strictly unique.")
    except Exception as e:
        errors.append(f"Failed to parse trans_id as integer: {e}")

    # Check timestamp
    try:
        parsed_ts = pd.to_datetime(df_copy["timestamp"], errors="coerce")
        if parsed_ts.isna().any():
            invalid_ts_count = int(parsed_ts.isna().sum())
            errors.append(f"Found {invalid_ts_count} unparseable timestamps. Format must be YYYY-MM-DD HH:MM:SS.")
        else:
            df_copy["timestamp"] = parsed_ts
    except Exception as e:
        errors.append(f"Timestamp parsing error: {e}")

    # Check amount
    try:
        df_copy["amount"] = pd.to_numeric(df_copy["amount"], errors="raise").astype("float64")
        if (df_copy["amount"] < 0).any():
            neg_count = int((df_copy["amount"] < 0).sum())
            errors.append(f"Detected {neg_count} negative transaction amount(s). Amounts must be non-negative.")
    except Exception as e:
        errors.append(f"Failed to cast amount to floating-point number: {e}")

    # Check merchant_category
    bad_cats = df_copy[~df_copy["merchant_category"].astype(str).isin(VALID_CATEGORIES)]["merchant_category"].unique()
    if len(bad_cats) > 0:
        errors.append(f"Invalid merchant_category values: {list(bad_cats)}. Expected subset of {sorted(VALID_CATEGORIES)}")

    # Check card_type
    bad_cards = df_copy[~df_copy["card_type"].astype(str).isin(VALID_CARD_TYPES)]["card_type"].unique()
    if len(bad_cards) > 0:
        errors.append(f"Invalid card_type values: {list(bad_cards)}. Expected subset of {sorted(VALID_CARD_TYPES)}")

    # Check is_fraud if present
    if "is_fraud" in df_copy.columns:
        try:
            df_copy["is_fraud"] = pd.to_numeric(df_copy["is_fraud"], errors="raise").astype("int64")
            unique_labels = set(df_copy["is_fraud"].unique())
            if not unique_labels.issubset({0, 1}):
                errors.append(f"is_fraud must be binary (0 or 1), but found values: {unique_labels}")
        except Exception as e:
            errors.append(f"Failed to cast is_fraud to binary integer: {e}")

    is_valid = len(errors) == 0
    return is_valid, errors, df_copy


def validate_single_transaction(record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate a single transaction dictionary (for single-record scoring / API input)."""
    df_single = pd.DataFrame([record])
    has_target = "is_fraud" in record and record["is_fraud"] is not None
    is_valid, errors, validated_df = validate_transaction_dataframe(
        df_single, require_target=has_target, allow_extra_columns=True
    )
    if is_valid:
        return True, [], validated_df.iloc[0].to_dict()
    return False, errors, record


def compute_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute an exploratory data quality and health diagnostic overview."""
    report: Dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "missing_values_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "total_missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows_exact": int(df.duplicated().sum()),
        "data_quality_warnings": [],
    }

    if "trans_id" in df.columns:
        report["duplicate_trans_id_count"] = int(df["trans_id"].duplicated().sum())
        if report["duplicate_trans_id_count"] > 0:
            report["data_quality_warnings"].append(
                f"Dataset contains {report['duplicate_trans_id_count']} duplicate trans_id values."
            )

    if "amount" in df.columns:
        amt_numeric = pd.to_numeric(df["amount"], errors="coerce")
        q25 = float(amt_numeric.quantile(0.25))
        q75 = float(amt_numeric.quantile(0.75))
        iqr = q75 - q25
        report["amount_summary"] = {
            "min": round(float(amt_numeric.min()), 2),
            "q25": round(q25, 2),
            "median": round(float(amt_numeric.median()), 2),
            "mean": round(float(amt_numeric.mean()), 2),
            "q75": round(q75, 2),
            "max": round(float(amt_numeric.max()), 2),
            "std": round(float(amt_numeric.std()), 2),
            "skewness": round(float(amt_numeric.skew()), 3),
            "iqr": round(iqr, 2),
        }
        if report["amount_summary"]["skewness"] > 1.5:
            report["data_quality_warnings"].append(
                f"Amount distribution is heavily right-skewed (skewness={report['amount_summary']['skewness']}). Log1p or robust scaling is strongly recommended."
            )

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if not ts.isna().all():
            min_ts = ts.min()
            max_ts = ts.max()
            span_days = round((max_ts - min_ts).total_seconds() / 86400.0, 2)
            report["timestamp_range"] = {
                "min": str(min_ts),
                "max": str(max_ts),
                "timespan_days": span_days,
            }

    if "is_fraud" in df.columns:
        label_counts = df["is_fraud"].value_counts().sort_index().to_dict()
        fraud_count = int(label_counts.get(1, 0))
        legit_count = int(label_counts.get(0, 0))
        total = fraud_count + legit_count
        fraud_rate = fraud_count / total if total > 0 else 0.0
        report["class_distribution"] = {
            "legitimate_count": legit_count,
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_rate, 4),
            "fraud_percentage": f"{fraud_rate * 100:.2f}%",
            "class_ratio": f"{legit_count}:{fraud_count}",
        }
        if fraud_count < 20:
            report["data_quality_warnings"].append(
                f"Severe minority class sparsity: only {fraud_count} fraud examples exist. Held-out test folds may contain <= 1 positive example, limiting statistical confidence."
            )

    if "merchant_category" in df.columns:
        report["merchant_category_counts"] = df["merchant_category"].value_counts().to_dict()

    if "card_type" in df.columns:
        report["card_type_counts"] = df["card_type"].value_counts().to_dict()

    return report
