"""Smoke tests for module imports and FastAPI REST endpoints."""
import pytest
from api.main import (
    BatchTransactionInput,
    TransactionInput,
    health_check,
    list_models,
    predict_batch,
    predict_single,
    read_audit_logs,
)


def test_api_health_endpoint():
    data = health_check()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "disclaimer" in data


def test_api_models_endpoint():
    data = list_models()
    assert "models" in data
    assert len(data["models"]) >= 1


def test_api_predict_single_endpoint():
    payload = TransactionInput(
        trans_id=999,
        timestamp="2024-01-05 14:30:00",
        card_token="CARD_XXXX1234",
        amount=250.00,
        merchant_category="Online",
        merchant_id="MER0045",
        card_type="Credit",
        model_name="Isolation Forest"
    )
    result = predict_single(payload)
    assert result["is_valid"] is True
    assert "anomaly_score" in result
    assert "risk_label" in result
    assert "disclaimer" in result


def test_api_predict_batch_endpoint():
    payload = BatchTransactionInput(
        transactions=[
            TransactionInput(
                trans_id=991,
                timestamp="2024-01-05 10:00:00",
                card_token="CARD_XXXX1111",
                amount=45.0,
                merchant_category="Grocery",
                merchant_id="MER0010",
                card_type="Debit",
            ),
            TransactionInput(
                trans_id=992,
                timestamp="2024-01-05 02:00:00",
                card_token="CARD_XXXX2222",
                amount=3200.0,
                merchant_category="Electronics",
                merchant_id="MER0020",
                card_type="Credit",
            )
        ],
        model_name="Isolation Forest"
    )
    result = predict_batch(payload)
    assert result["total_records"] == 2
    assert "flagged_for_review" in result
    assert len(result["results"]) == 2


def test_api_audit_logs_endpoint():
    data = read_audit_logs(limit=5)
    assert "logs" in data
