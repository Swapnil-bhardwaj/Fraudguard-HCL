"""Unit tests for feature engineering, leakage prevention, and temporal representations."""
import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_features_df():
    return pd.DataFrame([
        {
            "trans_id": 1,
            "timestamp": "2024-01-01 02:30:00",
            "card_token": "CARD_XXXX1111",
            "amount": 100.0,
            "merchant_category": "Online",
            "is_fraud": 0,
            "merchant_id": "MER0001",
            "card_type": "Credit",
        },
        {
            "trans_id": 2,
            "timestamp": "2024-01-01 14:45:00",
            "card_token": "CARD_XXXX1111",
            "amount": 250.0,
            "merchant_category": "Electronics",
            "is_fraud": 1,
            "merchant_id": "MER0001",
            "card_type": "Credit",
        },
        {
            "trans_id": 3,
            "timestamp": "2024-01-02 01:15:00",
            "card_token": "CARD_XXXX2222",
            "amount": 45.0,
            "merchant_category": "Grocery",
            "is_fraud": 0,
            "merchant_id": "MER0002",
            "card_type": "Debit",
        },
    ])


def test_leakage_guard_removes_target_and_identifiers(sample_features_df):
    fe = FeatureEngineer()
    out = fe.fit_transform(sample_features_df)
    
    # Critical leakage assertions
    assert "is_fraud" not in out.columns, "CRITICAL ERROR: is_fraud target leaked into feature matrix X!"
    assert "trans_id" not in out.columns, "Identifier trans_id must be excluded from features."
    assert "card_token" not in out.columns, "Raw card_token must not be used as a direct predictive feature."


def test_cyclical_temporal_features(sample_features_df):
    fe = FeatureEngineer()
    out = fe.fit_transform(sample_features_df)
    
    assert "hour_sin" in out.columns
    assert "hour_cos" in out.columns
    assert "is_night" in out.columns
    
    # Check bounded range [-1, 1]
    assert (out["hour_sin"] >= -1.0).all() and (out["hour_sin"] <= 1.0).all()
    assert (out["hour_cos"] >= -1.0).all() and (out["hour_cos"] <= 1.0).all()
    
    # Row 0 (02:30) is late night -> is_night should be 1
    assert out.loc[0, "is_night"] == 1
    # Row 1 (14:45) is afternoon -> is_night should be 0
    assert out.loc[1, "is_night"] == 0


def test_log1p_amount_transformation(sample_features_df):
    fe = FeatureEngineer()
    out = fe.fit_transform(sample_features_df)
    
    assert "log1p_amount" in out.columns
    expected_log = np.log1p(sample_features_df["amount"].values)
    np.testing.assert_allclose(out["log1p_amount"].values, expected_log, rtol=1e-5)


def test_card_velocity_temporal_integrity(sample_features_df):
    fe = FeatureEngineer(use_card_velocity=True)
    out = fe.fit_transform(sample_features_df)
    
    assert "card_prior_trans_count" in out.columns
    # Row 0 is the first transaction for CARD_XXXX1111 -> prior count must be 0
    assert out.loc[0, "card_prior_trans_count"] == 0
    # Row 1 is the second transaction for CARD_XXXX1111 -> prior count must be 1
    assert out.loc[1, "card_prior_trans_count"] == 1
    # Row 2 is the first transaction for CARD_XXXX2222 -> prior count must be 0
    assert out.loc[2, "card_prior_trans_count"] == 0
