"""Unit tests for data schema validation and error reporting."""
import pandas as pd
import pytest
from src.data_validation import (
    compute_data_quality_report,
    validate_single_transaction,
    validate_transaction_dataframe,
)


def sample_valid_df():
    return pd.DataFrame([
        {
            "trans_id": 1,
            "timestamp": "2024-01-01 12:00:00",
            "card_token": "CARD_XXXX1111",
            "amount": 50.25,
            "merchant_category": "Grocery",
            "is_fraud": 0,
            "merchant_id": "MER0001",
            "card_type": "Debit",
        },
        {
            "trans_id": 2,
            "timestamp": "2024-01-01 13:00:00",
            "card_token": "CARD_XXXX2222",
            "amount": 1200.00,
            "merchant_category": "Electronics",
            "is_fraud": 1,
            "merchant_id": "MER0002",
            "card_type": "Credit",
        },
    ])


def test_valid_dataframe_passes():
    df = sample_valid_df()
    is_valid, errors, validated_df = validate_transaction_dataframe(df, require_target=True)
    assert is_valid is True
    assert len(errors) == 0
    assert len(validated_df) == 2


def test_missing_required_column_fails():
    df = sample_valid_df().drop(columns=["amount"])
    is_valid, errors, _ = validate_transaction_dataframe(df, require_target=True)
    assert is_valid is False
    assert any("amount" in e for e in errors)


def test_negative_amount_fails():
    df = sample_valid_df()
    df.loc[0, "amount"] = -15.0
    is_valid, errors, _ = validate_transaction_dataframe(df, require_target=True)
    assert is_valid is False
    assert any("negative" in e.lower() for e in errors)


def test_invalid_category_fails():
    df = sample_valid_df()
    df.loc[0, "merchant_category"] = "Automotive"
    is_valid, errors, _ = validate_transaction_dataframe(df, require_target=True)
    assert is_valid is False
    assert any("merchant_category" in e for e in errors)


def test_non_binary_is_fraud_fails():
    df = sample_valid_df()
    df.loc[0, "is_fraud"] = 5
    is_valid, errors, _ = validate_transaction_dataframe(df, require_target=True)
    assert is_valid is False
    assert any("binary" in e.lower() for e in errors)


def test_duplicate_trans_id_fails():
    df = sample_valid_df()
    df.loc[1, "trans_id"] = 1
    is_valid, errors, _ = validate_transaction_dataframe(df, require_target=True)
    assert is_valid is False
    assert any("duplicate" in e.lower() for e in errors)


def test_unparseable_timestamp_fails():
    df = sample_valid_df()
    df.loc[0, "timestamp"] = "Not-A-Date"
    is_valid, errors, _ = validate_transaction_dataframe(df, require_target=True)
    assert is_valid is False
    assert any("timestamp" in e.lower() for e in errors)


def test_validate_single_transaction_record():
    record = {
        "trans_id": 10,
        "timestamp": "2024-01-02 10:15:00",
        "card_token": "CARD_XXXX9999",
        "amount": 75.0,
        "merchant_category": "Restaurant",
        "merchant_id": "MER0005",
        "card_type": "Prepaid",
    }
    is_valid, errors, validated = validate_single_transaction(record)
    assert is_valid is True
    assert len(errors) == 0
    assert validated["amount"] == 75.0


def test_compute_data_quality_report():
    df = sample_valid_df()
    report = compute_data_quality_report(df)
    assert report["row_count"] == 2
    assert report["column_count"] == 8
    assert report["total_missing_cells"] == 0
    assert "class_distribution" in report
