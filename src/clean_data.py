"""Validate and normalize the supplied fraud-detection CSV without synthesizing records.

Run from the project root:
    python src/clean_data.py --input data/dataset.csv --output data/dataset_cleaned.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "trans_id", "timestamp", "card_token", "amount",
    "merchant_category", "is_fraud", "merchant_id", "card_type"
]
VALID_CATEGORIES = {"Healthcare", "Fuel", "Travel", "Grocery", "Online", "Electronics", "Restaurant"}
VALID_CARD_TYPES = {"Credit", "Debit", "Prepaid"}


def clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype={"card_token": "string", "merchant_id": "string"})
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(f"Schema mismatch. Missing={missing_columns}; extra={extra_columns}; expected={EXPECTED_COLUMNS}")
    df = df[EXPECTED_COLUMNS].copy()
    if df[EXPECTED_COLUMNS].isna().any().any():
        missing = df.isna().sum().to_dict()
        raise ValueError(f"Missing values found; review before imputation: {missing}")

    df["trans_id"] = pd.to_numeric(df["trans_id"], errors="raise").astype("int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["amount"] = pd.to_numeric(df["amount"], errors="raise").astype("float64")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="raise").astype("int64")
    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1")
    if (df["amount"] < 0).any():
        raise ValueError("Negative transaction amounts detected")
    if df["trans_id"].duplicated().any():
        raise ValueError("Duplicate trans_id values detected")
    if not df["merchant_category"].isin(VALID_CATEGORIES).all():
        bad = sorted(set(df.loc[~df["merchant_category"].isin(VALID_CATEGORIES), "merchant_category"]))
        raise ValueError(f"Unexpected merchant_category values: {bad}")
    if not df["card_type"].isin(VALID_CARD_TYPES).all():
        bad = sorted(set(df.loc[~df["card_type"].isin(VALID_CARD_TYPES), "card_type"]))
        raise ValueError(f"Unexpected card_type values: {bad}")
    # CSV format is stable and readable; no rows are added or removed.
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["amount"] = df["amount"].map(lambda x: f"{x:.2f}")
    df["is_fraud"] = df["is_fraud"].astype("int64")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, columns=EXPECTED_COLUMNS)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/dataset.csv", type=Path)
    p.add_argument("--output", default="data/dataset_cleaned.csv", type=Path)
    args = p.parse_args()
    df = clean_data(args.input, args.output)
    print(f"Saved {len(df):,} rows × {len(df.columns)} columns to {args.output}")
    print("Label counts:", df["is_fraud"].value_counts().sort_index().to_dict())
    print("Exact duplicate rows:", int(df.duplicated().sum()))

if __name__ == "__main__":
    main()
