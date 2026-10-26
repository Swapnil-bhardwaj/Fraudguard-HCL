"""Unit tests for Isolation Forest, LOF, Autoencoder, and Supervised Baselines with SMOTE."""
import numpy as np
import pytest
from src.train_models import (
    train_autoencoder,
    train_isolation_forest,
    train_local_outlier_factor,
    train_supervised_baseline,
)


@pytest.fixture
def synthetic_matrices():
    np.random.seed(42)
    # 50 normal samples centered at 0, 4 outlier/fraud samples centered at 5
    X_normal = np.random.normal(0.0, 1.0, size=(50, 10))
    X_fraud = np.random.normal(5.0, 1.0, size=(4, 10))
    X = np.vstack([X_normal, X_fraud])
    y = np.array([0] * 50 + [1] * 4)
    return X, y


def test_isolation_forest_training_and_threshold(synthetic_matrices):
    X, _ = synthetic_matrices
    model, thresh = train_isolation_forest(X, contamination=0.08, n_estimators=50, random_seed=42)
    assert model is not None
    assert isinstance(thresh, float)
    
    # Inverted score: higher is more anomalous
    scores = -1.0 * model.score_samples(X)
    assert len(scores) == len(X)
    # The outliers (last 4 rows) should have higher anomaly scores than average normal rows
    assert np.mean(scores[-4:]) > np.mean(scores[:50])


def test_local_outlier_factor_novelty(synthetic_matrices):
    X, _ = synthetic_matrices
    X_train = X[:45]
    X_test = X[45:]
    
    # Must use novelty=True for out-of-sample prediction
    lof, thresh = train_local_outlier_factor(X_train, n_neighbors=10, contamination=0.05)
    assert lof.novelty is True
    
    # Test scoring on unseen data
    scores = -1.0 * lof.score_samples(X_test)
    assert len(scores) == len(X_test)


def test_mlp_autoencoder(synthetic_matrices):
    X, y = synthetic_matrices
    X_train_legit = X[y == 0]
    
    ae, thresh = train_autoencoder(
        X_train_legit,
        hidden_layer_sizes=(8, 4, 8),
        threshold_percentile=95.0,
        random_seed=42
    )
    assert ae.is_fitted is True
    assert thresh > 0.0
    
    # Test scoring on full set
    mse_scores = ae.score_samples(X)
    assert len(mse_scores) == len(X)
    # Fraud instances should incur higher reconstruction MSE error
    assert np.mean(mse_scores[-4:]) > np.mean(mse_scores[:50])


def test_supervised_baseline_with_smote(synthetic_matrices):
    X, y = synthetic_matrices
    clf, thresh, meta = train_supervised_baseline(
        X, y, use_smote=True, model_type="LogisticRegression", random_seed=42
    )
    assert meta["smote_applied"] is True
    assert meta["k_neighbors"] <= 3
    assert thresh == 0.50
    proba = clf.predict_proba(X)[:, 1]
    assert len(proba) == len(X)


def test_supervised_baseline_skips_smote_when_minority_under_two():
    X = np.random.randn(20, 5)
    y = np.array([0] * 19 + [1] * 1) # Only 1 minority instance
    
    clf, thresh, meta = train_supervised_baseline(
        X, y, use_smote=True, model_type="LogisticRegression", random_seed=42
    )
    # Must gracefully fall back to class weighting rather than crashing
    assert meta["smote_applied"] is False
    assert "Skipping SMOTE" in meta["smote_warning"]
