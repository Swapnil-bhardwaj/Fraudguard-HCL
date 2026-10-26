"""Preprocessing pipelines fitted exclusively on training data."""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

NUMERIC_FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_night",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "log1p_amount", "merchant_freq", "card_prior_trans_count"
]

CATEGORICAL_FEATURES = [
    "merchant_category", "card_type"
]


def create_preprocessor(scaler_type: str = "robust") -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that scales numeric features
    and one-hot encodes categorical features with robust handling of unknown categories.
    """
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


class TransactionPreprocessor:
    """
    End-to-end preprocessing wrapper managing feature transformation and scikit-learn pipeline.
    Ensures strict fit-on-train discipline.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        self.transformer: ColumnTransformer = create_preprocessor(scaler_type=scaler_type)
        self.is_fitted: bool = False
        self.output_feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TransactionPreprocessor":
        """Fit scaler and one-hot encoder on training data only."""
        self.transformer.fit(X)
        self.is_fitted = True

        # Extract output column names
        cat_encoder: OneHotEncoder = self.transformer.named_transformers_["cat"]
        cat_cols = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.output_feature_names_ = NUMERIC_FEATURES + cat_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features using parameters learned during fit()."""
        if not self.is_fitted:
            raise RuntimeError("TransactionPreprocessor must be fitted on training fold before transform().")
        return self.transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit on X and return transformed matrix."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return the names of transformed features."""
        return list(self.output_feature_names_)
