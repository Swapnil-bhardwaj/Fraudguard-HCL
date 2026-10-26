"""Unit tests for metric computation and imbalanced evaluation edge cases."""
import numpy as np
import pytest
from src.evaluation import compute_comprehensive_metrics


def test_standard_metrics_calculation():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9])
    threshold = 0.5
    
    m = compute_comprehensive_metrics(y_true, scores, threshold, latency_ms=1.5, model_name="TestModel")
    assert m["total_evaluated"] == 6
    assert m["actual_fraud_count"] == 2
    assert m["actual_legit_count"] == 4
    # Predictions: [0, 0, 0, 1, 1, 1]
    # TP: 2, FP: 1, TN: 3, FN: 0
    assert m["true_positives"] == 2
    assert m["false_positives"] == 1
    assert m["true_negatives"] == 3
    assert m["false_negatives"] == 0
    assert m["precision"] == round(2 / 3, 4)
    assert m["recall"] == 1.0
    assert m["pr_auc"] is not None
    assert m["roc_auc"] is not None


def test_single_class_edge_case_positives_zero():
    # Only negatives (0s), zero positive cases (1s)
    y_true = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = 0.25
    
    # Must NOT raise ValueError; handles edge cases gracefully
    m = compute_comprehensive_metrics(y_true, scores, threshold, latency_ms=1.0)
    assert m["actual_fraud_count"] == 0
    assert m["pr_auc"] is None
    assert "undefined" in m["pr_auc_note"].lower()
    assert m["roc_auc"] is None
    assert "undefined" in m["roc_auc_note"].lower()
    assert m["confusion_matrix"] == [[2, 2], [0, 0]]


def test_sparse_minority_warning():
    y_true = np.array([0] * 99 + [1] * 1) # 1 fraud
    scores = np.random.rand(100)
    m = compute_comprehensive_metrics(y_true, scores, threshold=0.5)
    assert m["reliability_warning"] is not None
    assert "Caution" in m["reliability_warning"]
