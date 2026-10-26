"""Feature engineering with strict target leakage prevention and temporal integrity."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Transforms raw transaction records into predictive numeric and categorical feature matrices.
    
    Guarantees:
    - Never includes 'is_fraud' in feature matrix X.
    - Excludes arbitrary identifiers 'trans_id' and raw 'card_token'.
    - Fits frequency encodings strictly on training fold X_train.
    - Applies cyclical transformations to hour-of-day.
    - Computes non-leaking prior transaction counts if card token history is enabled.
    """

    def __init__(self, use_card_velocity: bool = True):
        self.use_card_velocity = use_card_velocity
        self.merchant_freq_map_: Dict[str, float] = {}
        self.is_fitted_: bool = False
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit training-only feature parameters (e.g. merchant_id frequency map)."""
        df_copy = df.copy()
        
        # Calculate merchant_id frequency on training fold only
        if "merchant_id" in df_copy.columns:
            m_counts = df_copy["merchant_id"].value_counts(normalize=True).to_dict()
            self.merchant_freq_map_ = {str(k): float(v) for k, v in m_counts.items()}
        
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive domain features from validated transaction records.
        Returns a DataFrame ready for ColumnTransformer preprocessing.
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureEngineer must be fitted on training data before calling transform().")

        df_out = pd.DataFrame(index=df.index)

        # 1. Temporal feature engineering
        ts = pd.to_datetime(df["timestamp"])
        hour = ts.dt.hour
        day_of_week = ts.dt.dayofweek
        
        df_out["hour"] = hour
        df_out["day_of_week"] = day_of_week
        df_out["is_weekend"] = (day_of_week >= 5).astype(int)
        df_out["is_night"] = ((hour >= 23) | (hour < 5)).astype(int)
        
        # Cyclical continuous representations of time (preserves 23:59 -> 00:01 circularity)
        df_out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df_out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        df_out["day_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        df_out["day_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

        # 2. Amount transformation
        # Raw amounts have heavy positive skew; log1p stabilizes variance
        amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df_out["amount"] = amount
        df_out["log1p_amount"] = np.log1p(amount)

        # 3. Categorical variables to be one-hot encoded
        df_out["merchant_category"] = df["merchant_category"].astype(str)
        df_out["card_type"] = df["card_type"].astype(str)

        # 4. Merchant ID representation
        # Unseen merchants receive 0.0 frequency
        if "merchant_id" in df.columns:
            df_out["merchant_freq"] = df["merchant_id"].astype(str).map(self.merchant_freq_map_).fillna(0.0)
        else:
            df_out["merchant_freq"] = 0.0

        # 5. Non-leaking Card Velocity (prior transactions count for card)
        # Prevents future information leakage: sorts strictly by timestamp and counts only prior rows
        if self.use_card_velocity and "card_token" in df.columns and "timestamp" in df.columns:
            # Sort chronologically, groupby card_token, and use cumcount (which equals number of prior transactions)
            temp = pd.DataFrame({
                "orig_idx": df.index,
                "card_token": df["card_token"].astype(str),
                "timestamp": ts
            }).sort_values("timestamp")
            temp["card_prior_trans_count"] = temp.groupby("card_token").cumcount()
            df_out["card_prior_trans_count"] = temp.sort_values("orig_idx")["card_prior_trans_count"].values
        else:
            df_out["card_prior_trans_count"] = 0

        # CRITICAL SAFETY: Ensure 'is_fraud', 'trans_id', and raw 'card_token' are NOT in features
        for forbidden in ["is_fraud", "trans_id", "card_token"]:
            if forbidden in df_out.columns:
                df_out.drop(columns=[forbidden], inplace=True)

        self.feature_names_ = list(df_out.columns)
        return df_out

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit parameters on df and return transformed DataFrame."""
        return self.fit(df, y).transform(df)
