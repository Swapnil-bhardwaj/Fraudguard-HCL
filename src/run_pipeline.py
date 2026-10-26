"""End-to-end training, evaluation, model persistence, and report generation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from typing import Dict, List
import pandas as pd
import numpy as np

from src.data_validation import validate_transaction_dataframe
from src.feature_engineering import FeatureEngineer
from src.preprocessing import TransactionPreprocessor
from src.train_models import (
    ModelPipeline,
    create_train_test_split,
    train_autoencoder,
    train_isolation_forest,
    train_local_outlier_factor,
    train_supervised_baseline,
)
from src.evaluation import (
    compute_comprehensive_metrics,
    plot_confusion_matrix,
    plot_metrics_comparison,
)
from src.persistence import save_model_pipeline, init_audit_db
from src.utils import NumpyJSONEncoder, benchmark_latency, setup_logger, set_seed

logger = setup_logger("pipeline_runner")


def run_pipeline(
    dataset_path: Path = Path("data/dataset_cleaned.csv"),
    models_dir: Path = Path("models"),
    outputs_dir: Path = Path("outputs"),
    test_size: float = 0.2,
    seed: int = 42,
    contamination: float = 0.02,
) -> Dict[str, pd.DataFrame]:
    """Execute complete end-to-end ML lifecycle."""
    set_seed(seed)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = outputs_dir / "figures"
    metrics_dir = outputs_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    init_audit_db(outputs_dir / "audit.db")

    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # 1. Validation
    is_valid, errors, validated_df = validate_transaction_dataframe(df, require_target=True)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {errors}")
    logger.info(f"Dataset validated successfully: {len(validated_df)} rows.")

    # 2. Partition Data
    splits = create_train_test_split(
        validated_df,
        test_size=test_size,
        random_seed=seed,
        stratify=True
    )

    # 3. Fit Feature Engineering and Preprocessing on Train fold only
    logger.info("Fitting FeatureEngineer and TransactionPreprocessor on training fold only...")
    fe = FeatureEngineer(use_card_velocity=True)
    X_train_fe = fe.fit_transform(splits.X_train_raw)
    X_test_fe = fe.transform(splits.X_test_raw)

    preprocessor = TransactionPreprocessor(scaler_type="robust")
    X_train_proc = preprocessor.fit_transform(X_train_fe)
    X_test_proc = preprocessor.transform(X_test_fe)

    y_train = splits.y_train.values
    y_test = splits.y_test.values

    # 4. Train Models
    pipelines: List[ModelPipeline] = []

    # Model 1: Isolation Forest
    logger.info("Training Isolation Forest...")
    iso_model, iso_thresh = train_isolation_forest(
        X_train_proc,
        contamination=contamination,
        random_seed=seed,
    )
    p_iso = ModelPipeline(
        model_name="Isolation Forest",
        model=iso_model,
        feature_engineer=fe,
        preprocessor=preprocessor,
        threshold=iso_thresh,
        metadata={"contamination": contamination, "n_estimators": 150},
    )
    pipelines.append(p_iso)

    # Model 2: Local Outlier Factor (novelty=True)
    logger.info("Training Local Outlier Factor (novelty=True)...")
    lof_model, lof_thresh = train_local_outlier_factor(
        X_train_proc,
        n_neighbors=20,
        contamination=contamination,
    )
    p_lof = ModelPipeline(
        model_name="Local Outlier Factor",
        model=lof_model,
        feature_engineer=fe,
        preprocessor=preprocessor,
        threshold=lof_thresh,
        metadata={"n_neighbors": 20, "contamination": contamination},
    )
    pipelines.append(p_lof)

    # Model 3: Autoencoder (Hourglass MLP)
    logger.info("Training Neural Autoencoder...")
    # Train predominantly on normal/legitimate transactions
    X_train_legit = X_train_proc[y_train == 0]
    ae_model, ae_thresh = train_autoencoder(
        X_train_legit,
        hidden_layer_sizes=(16, 8, 16),
        threshold_percentile=95.0,
        random_seed=seed,
    )
    p_ae = ModelPipeline(
        model_name="Autoencoder",
        model=ae_model,
        feature_engineer=fe,
        preprocessor=preprocessor,
        threshold=ae_thresh,
        metadata={"architecture": [16, 8, 16], "threshold_percentile": 95.0},
    )
    pipelines.append(p_ae)

    # Model 4: Supervised Baseline (Logistic Regression, class-weighted)
    logger.info("Training Supervised Baseline (Logistic Regression)...")
    lr_model, lr_thresh, lr_meta = train_supervised_baseline(
        X_train_proc,
        y_train,
        use_smote=False,
        model_type="LogisticRegression",
        random_seed=seed,
    )
    p_lr = ModelPipeline(
        model_name="Logistic Regression (Supervised)",
        model=lr_model,
        feature_engineer=fe,
        preprocessor=preprocessor,
        threshold=lr_thresh,
        metadata=lr_meta,
    )
    pipelines.append(p_lr)

    # Model 5: Supervised Baseline with SMOTE (inside fold only)
    logger.info("Training Supervised Baseline with SMOTE inside train fold...")
    smote_model, smote_thresh, smote_meta = train_supervised_baseline(
        X_train_proc,
        y_train,
        use_smote=True,
        model_type="LogisticRegression",
        random_seed=seed,
    )
    p_smote = ModelPipeline(
        model_name="Logistic Regression (SMOTE)",
        model=smote_model,
        feature_engineer=fe,
        preprocessor=preprocessor,
        threshold=smote_thresh,
        metadata=smote_meta,
    )
    pipelines.append(p_smote)

    # 5. Evaluate on Held-Out Test Set
    metrics_list: List[Dict] = []
    logger.info("Evaluating all models on held-out test set...")

    for p in pipelines:
        # Measure scoring latency
        _, latency_ms = benchmark_latency(p.score_processed_matrix, X_test_proc, iterations=30)
        
        # Anomaly / fraud score
        scores = p.score_processed_matrix(X_test_proc)
        
        # Calculate full metrics
        metrics = compute_comprehensive_metrics(
            y_true=y_test,
            scores=scores,
            threshold=p.threshold,
            latency_ms=latency_ms,
            model_name=p.model_name,
        )
        metrics_list.append(metrics)

        # Plot confusion matrix
        slug = p.model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            model_name=p.model_name,
            save_path=figures_dir / f"confusion_matrix_{slug}.png",
        )

        # Persist model pipeline
        save_model_pipeline(p, save_dir=models_dir, version="v1")

    # 6. Save Metrics & Comparison
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(metrics_dir / "model_comparison.csv", index=False)

    summary_payload = {
        "dataset_evaluated": str(dataset_path),
        "total_records": len(df),
        "train_size": len(splits.X_train_raw),
        "train_legit": splits.train_legit_count,
        "train_fraud": splits.train_fraud_count,
        "test_size": len(splits.X_test_raw),
        "test_legit": splits.test_legit_count,
        "test_fraud": splits.test_fraud_count,
        "random_seed": seed,
        "contamination": contamination,
        "models_evaluated": metrics_list,
    }
    (metrics_dir / "evaluation_results.json").write_text(
        json.dumps(summary_payload, indent=2, cls=NumpyJSONEncoder),
        encoding="utf-8"
    )

    # Save summary bar chart
    plot_metrics_comparison(metrics_df, save_path=figures_dir / "model_metrics_comparison.png")

    logger.info("Pipeline execution complete! Summary of results:")
    display_cols = ["model_name", "precision", "recall", "f1_score", "pr_auc", "roc_auc", "false_positive_rate", "latency_ms"]
    print("\n" + "="*80)
    print("EVALUATION RESULTS ON HELD-OUT TEST DATA")
    print("="*80)
    print(metrics_df[display_cols].to_string(index=False))
    print("="*80 + "\n")

    return {"metrics_df": metrics_df, "summary": summary_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset_cleaned.csv", type=Path)
    parser.add_argument("--models-dir", default="models", type=Path)
    parser.add_argument("--outputs-dir", default="outputs", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--contamination", default=0.02, type=float)
    args = parser.parse_args()

    run_pipeline(
        dataset_path=args.dataset,
        models_dir=args.models_dir,
        outputs_dir=args.outputs_dir,
        seed=args.seed,
        contamination=args.contamination,
    )


if __name__ == "__main__":
    main()
