"""Exploratory data analysis for Credit Card Fraud Detection Using Anomaly Detection.

Run from the project root:
    python src/eda.py --input data/dataset_cleaned.csv --output-dir outputs

Creates four PNG visualisations and a JSON summary. Uses only the supplied CSV and
never fabricates transactions or model metrics.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED = ["trans_id", "timestamp", "card_token", "amount", "merchant_category", "is_fraud", "merchant_id", "card_type"]


def run_eda(input_path: Path, output_dir: Path) -> dict:
    df = pd.read_csv(input_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
    if df[["timestamp", "amount", "is_fraud"]].isna().any().any():
        raise ValueError("Timestamp, amount, or label parse errors exist. Run clean_data.py and inspect source rows.")
    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1")
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # 1. Label counts
    counts = df["is_fraud"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(["Legitimate (0)", "Fraud-labelled (1)"], counts.values)
    ax.set_title("Class Distribution in Supplied Dataset")
    ax.set_ylabel("Number of transactions")
    ax.bar_label(bars, labels=[f"{int(v):,} ({v/len(df):.1%})" for v in counts.values], padding=3)
    fig.tight_layout(); fig.savefig(figures / "class_distribution.png", dpi=160); plt.close(fig)

    # 2. Transaction amount by known label
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [df.loc[df["is_fraud"] == c, "amount"].values for c in [0, 1]]
    ax.boxplot(series, tick_labels=["Legitimate (0)", "Fraud-labelled (1)"], showfliers=True)
    ax.set_title("Transaction Amount by Label")
    ax.set_ylabel("Amount (currency unspecified)")
    ax.set_yscale("symlog", linthresh=100)
    fig.tight_layout(); fig.savefig(figures / "amount_by_label.png", dpi=160); plt.close(fig)

    # 3. Categories: total records and fraud-labelled count
    cats = df.groupby("merchant_category").agg(total=("trans_id", "count"), fraud=("is_fraud", "sum")).sort_values("total", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(cats))
    ax.bar(x, cats["total"], label="All transactions")
    ax.bar(x, cats["fraud"], label="Fraud-labelled rows")
    ax.set_xticks(list(x), cats.index, rotation=28, ha="right")
    ax.set_title("Transactions by Merchant Category")
    ax.set_ylabel("Number of transactions")
    ax.legend()
    fig.tight_layout(); fig.savefig(figures / "merchant_category_counts.png", dpi=160); plt.close(fig)

    # 4. Daily volume and labels
    daily = df.set_index("timestamp").resample("D").agg(total=("trans_id", "count"), fraud=("is_fraud", "sum"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(daily.index, daily["total"], marker="o", label="All transactions")
    ax.plot(daily.index, daily["fraud"], marker="o", label="Fraud-labelled rows")
    ax.set_title("Transaction Counts over Time")
    ax.set_ylabel("Number of transactions")
    ax.set_xlabel("Timestamp (day)")
    ax.legend()
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(figures / "transactions_over_time.png", dpi=160); plt.close(fig)

    fraud_count = int(df["is_fraud"].sum())
    category = df.groupby("merchant_category").agg(total=("trans_id", "count"), fraud=("is_fraud", "sum"))
    summary = {
        "input_file": str(input_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_values_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_trans_id_count": int(df["trans_id"].duplicated().sum()),
        "fraud_label_counts": {str(k): int(v) for k, v in df["is_fraud"].value_counts().sort_index().items()},
        "fraud_label_rate": fraud_count / len(df) if len(df) else None,
        "timestamp_min": df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_max": df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "amount_summary": {"min": float(df["amount"].min()), "max": float(df["amount"].max()), "mean": float(df["amount"].mean()), "median": float(df["amount"].median())},
        "merchant_category_counts": {str(k): {"total": int(v["total"]), "fraud_labelled": int(v["fraud"])} for k, v in category.iterrows()},
        "card_type_counts": {str(k): int(v) for k, v in df["card_type"].value_counts().items()},
        "figures": ["class_distribution.png", "amount_by_label.png", "merchant_category_counts.png", "transactions_over_time.png"],
        "interpretation_warning": "The file has only six fraud-labelled rows; per-category fraud rates and model metrics are unstable. No model has been trained by this EDA script. Currency is not specified."
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("data/dataset_cleaned.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = p.parse_args()
    result = run_eda(args.input, args.output_dir)
    print(json.dumps({k: result[k] for k in ["rows", "columns", "fraud_label_counts", "fraud_label_rate", "exact_duplicate_rows", "timestamp_min", "timestamp_max"]}, indent=2))
    print(f"Saved figures under {args.output_dir / 'figures'} and summary to {args.output_dir / 'eda_summary.json'}")

if __name__ == "__main__":
    main()
