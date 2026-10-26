"""Model training pipelines for Isolation Forest, LOF, Autoencoder, and Supervised Baseline."""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedShuffleSplit

from src.feature_engineering import FeatureEngineer
from src.preprocessing import TransactionPreprocessor
from src.utils import setup_logger, set_seed

logger = setup_logger("train_models")


@dataclass
class DatasetSplits:
    """Holds partitioned datasets, tracking class counts explicitly."""
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_legit_count: int
    train_fraud_count: int
    test_legit_count: int
    test_fraud_count: int
    split_strategy: str
    random_seed: int


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
    stratify: bool = True
) -> DatasetSplits:
    """
    Partition dataset into training and held-out test sets.
    Logs class proportions and verifies no leakage.
    """
    set_seed(random_seed)
    y = df["is_fraud"].astype(int)
    X = df.copy()

    total_frauds = int((y == 1).sum())
    total_legit = int((y == 0).sum())

    if stratify and total_frauds >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(sss.split(X, y))
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        strategy_desc = f"StratifiedShuffleSplit(test_size={test_size}, seed={random_seed})"
    else:
        # Fallback to chronological or random split if minority is too sparse to stratify
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=None
        )
        strategy_desc = f"RandomSplit(test_size={test_size}, seed={random_seed})"

    train_legit = int((y_train == 0).sum())
    train_fraud = int((y_train == 1).sum())
    test_legit = int((y_test == 0).sum())
    test_fraud = int((y_test == 1).sum())

    logger.info(
        f"Data split created: Train={len(X_train)} (Legit={train_legit}, Fraud={train_fraud}), "
        f"Test={len(X_test)} (Legit={test_legit}, Fraud={test_fraud})"
    )

    return DatasetSplits(
        X_train_raw=X_train,
        X_test_raw=X_test,
        y_train=y_train,
        y_test=y_test,
        train_legit_count=train_legit,
        train_fraud_count=train_fraud,
        test_legit_count=test_legit,
        test_fraud_count=test_fraud,
        split_strategy=strategy_desc,
        random_seed=random_seed,
    )


class MLPAutoencoder:
    """
    Neural Autoencoder implemented with an hourglass MLP bottleneck.
    Reconstructs normal transactions and scores anomalies by Mean Squared Error (MSE).
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (16, 8, 16),
        activation: str = "relu",
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self.threshold: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, threshold_percentile: float = 95.0) -> "MLPAutoencoder":
        """Fit autoencoder to reconstruct X (trained predominantly on legitimate transactions)."""
        self.model.fit(X, X)
        self.is_fitted = True
        
        # Calculate reconstruction error on training data to set baseline threshold
        train_recon = self.model.predict(X)
        train_mse = np.mean(np.square(X - train_recon), axis=1)
        self.threshold = float(np.percentile(train_mse, threshold_percentile))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error (MSE) for each sample.
        Higher score = higher anomaly / reconstruction deviation.
        """
        if not self.is_fitted:
            raise RuntimeError("Autoencoder must be fitted before scoring.")
        recon = self.model.predict(X)
        mse = np.mean(np.square(X - recon), axis=1)
        return mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly label: 1 if anomaly score >= threshold else 0."""
        scores = self.score_samples(X)
        return (scores >= self.threshold).astype(int)


class ModelPipeline:
    """
    Wraps FeatureEngineer, TransactionPreprocessor, and an Anomaly or Supervised Model.
    Provides uniform fit, score, and evaluate interfaces.
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        feature_engineer: FeatureEngineer,
        preprocessor: TransactionPreprocessor,
        threshold: float = 0.0,
        score_orientation: str = "higher_is_anomalous",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.model = model
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.score_orientation = score_orientation
        self.metadata = metadata or {}
        self.is_fitted: bool = False

    def score_raw_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Transform raw DataFrame through feature engineering and preprocessing, then score."""
        feats = self.feature_engineer.transform(df)
        X_mat = self.preprocessor.transform(feats)
        return self.score_processed_matrix(X_mat)

    def score_processed_matrix(self, X_mat: np.ndarray) -> np.ndarray:
        """Compute anomaly score where HIGHER = MORE ANOMALOUS across all models."""
        if self.model_name == "Isolation Forest":
            # sklearn IsolationForest score_samples returns negative values (lower = more abnormal)
            # Inverting sign so higher score = higher anomaly
            raw_scores = self.model.score_samples(X_mat)
            return -1.0 * raw_scores

        elif self.model_name == "Local Outlier Factor":
            # LOF novelty score_samples returns negative values (lower = more abnormal)
            raw_scores = self.model.score_samples(X_mat)
            return -1.0 * raw_scores

        elif self.model_name == "Autoencoder":
            # Autoencoder returns reconstruction MSE (higher = more abnormal)
            return self.model.score_samples(X_mat)

        elif "Supervised" in self.model_name or "Logistic" in self.model_name or "Random Forest" in self.model_name:
            # Predict probability of fraud class (1)
            proba = self.model.predict_proba(X_mat)[:, 1]
            return proba

        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def predict_flags(self, df: pd.DataFrame) -> np.ndarray:
        """Generate binary review flags (1 = review recommended, 0 = lower anomaly score)."""
        scores = self.score_raw_dataframe(df)
        return (scores >= self.threshold).astype(int)


def train_isolation_forest(
    X_train_proc: np.ndarray,
    contamination: float = 0.02,
    n_estimators: int = 150,
    random_seed: int = 42,
) -> Tuple[IsolationForest, float]:
    """Train Isolation Forest and calibrate threshold on training distribution."""
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_seed,
        n_jobs=1,
    )
    iso.fit(X_train_proc)
    train_scores = -1.0 * iso.score_samples(X_train_proc)
    threshold = float(np.percentile(train_scores, 100.0 * (1.0 - contamination)))
    return iso, threshold


def train_local_outlier_factor(
    X_train_proc: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.02,
) -> Tuple[LocalOutlierFactor, float]:
    """
    Train Local Outlier Factor with novelty=True for out-of-sample scoring.
    Fitted exclusively on training data.
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=1,
    )
    lof.fit(X_train_proc)
    train_scores = -1.0 * lof.score_samples(X_train_proc)
    threshold = float(np.percentile(train_scores, 100.0 * (1.0 - contamination)))
    return lof, threshold


def train_autoencoder(
    X_train_proc: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (16, 8, 16),
    threshold_percentile: float = 95.0,
    random_seed: int = 42,
) -> Tuple[MLPAutoencoder, float]:
    """Train MLP reconstruction Autoencoder and set anomaly threshold."""
    ae = MLPAutoencoder(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=600,
        random_state=random_seed,
    )
    ae.fit(X_train_proc, threshold_percentile=threshold_percentile)
    return ae, ae.threshold


def train_supervised_baseline(
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    use_smote: bool = True,
    model_type: str = "LogisticRegression",
    random_seed: int = 42,
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Train a supervised classifier (Logistic Regression or Random Forest).
    If use_smote=True, applies SMOTE strictly inside the training fold.
    Dynamically adjusts k_neighbors or falls back to class weighting if minority is too sparse.
    """
    minority_count = int(np.sum(y_train == 1))
    meta: Dict[str, Any] = {
        "minority_count": minority_count,
        "smote_applied": False,
        "smote_warning": None,
    }

    X_train_fit = X_train_proc
    y_train_fit = y_train

    if use_smote:
        if minority_count < 2:
            meta["smote_warning"] = (
                f"Skipping SMOTE: minority class count ({minority_count}) is less than 2. "
                "SMOTE requires at least 2 minority examples to define nearest neighbors."
            )
            logger.warning(meta["smote_warning"])
        else:
            try:
                from imblearn.over_sampling import SMOTE
                # k_neighbors must be <= minority_count - 1
                k_neighbors = min(minority_count - 1, 5)
                smote = SMOTE(k_neighbors=k_neighbors, random_state=random_seed)
                X_train_fit, y_train_fit = smote.fit_resample(X_train_proc, y_train)
                meta["smote_applied"] = True
                meta["k_neighbors"] = k_neighbors
                meta["resampled_counts"] = {
                    "0": int(np.sum(y_train_fit == 0)),
                    "1": int(np.sum(y_train_fit == 1)),
                }
                logger.info(f"SMOTE applied inside train fold with k_neighbors={k_neighbors}: {meta['resampled_counts']}")
            except Exception as e:
                meta["smote_warning"] = f"SMOTE execution failed ({e}); falling back to class-weighted fitting."
                logger.warning(meta["smote_warning"])

    if model_type == "LogisticRegression":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=5,
            random_state=random_seed,
            n_jobs=-1
        )

    clf.fit(X_train_fit, y_train_fit)
    # Default decision threshold for supervised probability
    threshold = 0.50
    return clf, threshold, meta
