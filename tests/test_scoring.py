"""Unit tests for inference scoring, thresholding, and risk labels."""
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer
from src.persistence import load_model_pipeline
from src.preprocessing import TransactionPreprocessor
from src.scoring import score_batch_dataframe, score_single_record
from src.train_models import ModelPipeline, train_isolation_forest


@pytest.fixture
def fitted_pipeline():
    df_train = pd.DataFrame([
        {"trans_id": 1, "timestamp": "2024-01-01 10:00:00", "card_token": "CARD_A", "amount": 25.0, "merchant_category": "Grocery", "merchant_id": "MER1", "card_type": "Debit"},
        {"trans_id": 2, "timestamp": "2024-01-01 11:00:00", "card_token": "CARD_B", "amount": 40.0, "merchant_category": "Fuel", "merchant_id": "MER2", "card_type": "Credit"},
        {"trans_id": 3, "timestamp": "2024-01-01 12:00:00", "card_token": "CARD_C", "amount": 60.0, "merchant_category": "Restaurant", "merchant_id": "MER3", "card_type": "Debit"},
        {"trans_id": 4, "timestamp": "2024-01-01 13:00:00", "card_token": "CARD_D", "amount": 35.0, "merchant_category": "Online", "merchant_id": "MER4", "card_type": "Credit"},
        {"trans_id": 5, "timestamp": "2024-01-01 14:00:00", "card_token": "CARD_E", "amount": 1500.0, "merchant_category": "Electronics", "merchant_id": "MER5", "card_type": "Credit"},
    ])
    fe = FeatureEngineer()
    X_fe = fe.fit_transform(df_train)
    pre = TransactionPreprocessor()
    X_proc = pre.fit_transform(X_fe)
    iso, thresh = train_isolation_forest(X_proc, contamination=0.20, n_estimators=50, random_seed=42)

    return ModelPipeline(
        model_name="Isolation Forest",
        model=iso,
        feature_engineer=fe,
        preprocessor=pre,
        threshold=thresh,
    )


def test_score_single_record(fitted_pipeline):
    record = {
        "trans_id": 99,
        "timestamp": "2024-01-02 03:15:00",
        "card_token": "CARD_XXXX1234",
        "amount": 2850.00,
        "merchant_category": "Electronics",
        "merchant_id": "MER99",
        "card_type": "Credit",
    }
    result = score_single_record(record, fitted_pipeline)
    assert result["is_valid"] is True
    assert "anomaly_score" in result
    assert "threshold" in result
    assert result["risk_label"] in ["Review recommended", "Lower anomaly score"]
    assert "disclaimer" in result
    assert "human_triage_action" in result


def test_score_batch_dataframe(fitted_pipeline):
    batch_df = pd.DataFrame([
        {"trans_id": 101, "timestamp": "2024-01-03 10:00:00", "card_token": "CARD_A", "amount": 20.0, "merchant_category": "Grocery", "merchant_id": "MER1", "card_type": "Debit"},
        {"trans_id": 102, "timestamp": "2024-01-03 02:00:00", "card_token": "CARD_B", "amount": 3500.0, "merchant_category": "Travel", "merchant_id": "MER2", "card_type": "Credit"},
    ])
    scored = score_batch_dataframe(batch_df, fitted_pipeline)
    assert len(scored) == 2
    assert "anomaly_score" in scored.columns
    assert "threshold" in scored.columns
    assert "flag" in scored.columns
    assert "risk_label" in scored.columns
    assert "neutral_indicator" in scored.columns
