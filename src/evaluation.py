"""Evaluation metrics computation, comparison tables, and diagnostic plots."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from pathlib import Path


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    latency_ms: float = 0.0,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Calculate full evaluation metrics for fraud anomaly detection.
    Guarantees no crashes on edge cases (e.g. single-class test fold).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)

    total_samples = len(y_true)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))

    # Confusion matrix extraction
    # Using labels=[0, 1] ensures 2x2 matrix even if one label is absent in batch
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Precision, Recall, F1 for the fraud class (1)
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    acc = float((tp + tn) / total_samples) if total_samples > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Average Precision / PR-AUC
    pr_auc: Optional[float] = None
    pr_auc_note: Optional[str] = None
    if positives > 0:
        try:
            pr_auc = float(average_precision_score(y_true, scores))
        except Exception as e:
            pr_auc_note = f"PR-AUC calculation exception: {e}"
    else:
        pr_auc_note = "PR-AUC undefined: held-out set contains zero positive (fraud) instances."

    # ROC-AUC
    roc_auc: Optional[float] = None
    roc_auc_note: Optional[str] = None
    if positives > 0 and negatives > 0:
        try:
            roc_auc = float(roc_auc_score(y_true, scores))
        except Exception as e:
            roc_auc_note = f"ROC-AUC calculation exception: {e}"
    else:
        roc_auc_note = (
            "ROC-AUC mathematically undefined: only one class is present in the evaluated sample "
            f"(positives={positives}, negatives={negatives})."
        )

    # Reliability warning for sparse evaluation sets
    reliability_warning = None
    if positives <= 2:
        reliability_warning = (
            f"Caution: Test set contains only {positives} fraud record(s). "
            "A single correct or missed prediction shifts recall between 0% and 100%. "
            "Metrics reflect an academic prototype demonstration, not production confidence."
        )

    return {
        "model_name": model_name,
        "total_evaluated": total_samples,
        "actual_fraud_count": positives,
        "actual_legit_count": negatives,
        "threshold": round(float(threshold), 5),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "frauds_missed": fn,
        "legit_flagged": fp,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "false_positive_rate": round(fpr, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "pr_auc_note": pr_auc_note,
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "roc_auc_note": roc_auc_note,
        "latency_ms": round(float(latency_ms), 3),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "reliability_warning": reliability_warning,
    }


def plot_confusion_matrix(cm: List[List[int]], model_name: str, save_path: Optional[Path] = None):
    """Render a clean 2x2 confusion matrix with counts and percentages."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    cm_arr = np.array(cm)
    cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center", color="black", fontsize=12, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1], ["Predicted Legit (0)", "Predicted Review (1)"])
    ax.set_yticks([0, 1], ["Actual Legit (0)", "Actual Fraud (1)"])
    ax.set_title(f"Confusion Matrix — {model_name}", pad=14)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    return fig


def plot_metrics_comparison(metrics_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Bar chart comparing Precision, Recall, and F1 across models."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(metrics_df))
    width = 0.25

    ax.bar(x - width, metrics_df["precision"], width, label="Precision", color="#2b5c8f")
    ax.bar(x, metrics_df["recall"], width, label="Recall", color="#e27c38")
    ax.bar(x + width, metrics_df["f1_score"], width, label="F1-Score", color="#2ca02c")

    ax.set_xticks(x, metrics_df["model_name"])
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Model Comparison on Held-Out Evaluation Set")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    return fig
