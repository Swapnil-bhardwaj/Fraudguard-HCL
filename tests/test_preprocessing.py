"""Unit tests for preprocessing and out-of-fold generalization."""
import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer
from src.preprocessing import TransactionPreprocessor


@pytest.fixture
def train_test_data():
    train_raw = pd.DataFrame([
        {"trans_id": 1, "timestamp": "2024-01-01 10:00:00", "card_token": "CARD_A", "amount": 50.0, "merchant_category": "Grocery", "merchant_id": "MER1", "card_type": "Debit"},
        {"trans_id": 2, "timestamp": "2024-01-01 12:00:00", "card_token": "CARD_B", "amount": 150.0, "merchant_category": "Online", "merchant_id": "MER2", "card_type": "Credit"},
        {"trans_id": 3, "timestamp": "2024-01-01 14:00:00", "card_token": "CARD_A", "amount": 300.0, "merchant_category": "Fuel", "merchant_id": "MER1", "card_type": "Prepaid"},
    ])
    
    # Test data contains an unseen category level in merchant_category ("Healthcare")
    test_raw = pd.DataFrame([
        {"trans_id": 4, "timestamp": "2024-01-02 10:00:00", "card_token": "CARD_C", "amount": 80.0, "merchant_category": "Healthcare", "merchant_id": "MER3", "card_type": "Debit"},
    ])
    return train_raw, test_raw


def test_fit_on_train_transforms_test_without_error(train_test_data):
    train_raw, test_raw = train_test_data
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(train_raw)
    X_test_fe = fe.transform(test_raw)

    preprocessor = TransactionPreprocessor(scaler_type="robust")
    X_train_proc = preprocessor.fit_transform(X_train_fe)
    
    # Test set transformation must succeed even with unseen category levels (handle_unknown='ignore')
    X_test_proc = preprocessor.transform(X_test_fe)

    assert X_train_proc.shape[0] == 3
    assert X_test_proc.shape[0] == 1
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert not np.isnan(X_test_proc).any(), "Preprocessed matrix contains unexpected NaNs."


def test_unfitted_transform_raises_error():
    preprocessor = TransactionPreprocessor()
    df = pd.DataFrame({"amount": [10.0]})
    with pytest.raises(RuntimeError):
        preprocessor.transform(df)
