"""Generate clearly labelled synthetic transaction rows strictly separated from dataset.csv.

Follows the prompt specifications in prompts/SYNTHETIC_DATA_GENERATION_PROMPT.txt:
- Generates 4,500 rows with trans_id from 501 to 5000.
- Timestamps span 2024-01-12 to 2024-02-15 (after original data).
- Simulates realistic fraud scenarios with realistic class overlap.
- Emits synthetic_transactions_only.csv and synthetic_data_manifest.json.
- Never alters dataset.csv.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd

VALID_CATEGORIES = ["Healthcare", "Fuel", "Travel", "Grocery", "Online", "Electronics", "Restaurant"]
VALID_CARD_TYPES = ["Credit", "Debit", "Prepaid"]


def generate_synthetic_data(
    num_rows: int = 4500,
    start_id: int = 501,
    start_time: str = "2024-01-12 00:05:00",
    seed: int = 42,
    output_csv: Path = Path("data/synthetic_transactions_only.csv"),
    output_manifest: Path = Path("data/synthetic_data_manifest.json"),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    current_dt = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    
    # Pool of simulated masked cards and merchants
    card_pool = [f"CARD_XXXX{rng.integers(1000, 9999)}" for _ in range(350)]
    merchant_pool = [f"MER{rng.integers(1, 500):04d}" for _ in range(250)]

    records = []
    
    # Category typical base amounts (lognormal parameters)
    cat_params = {
        "Grocery": (3.5, 0.6),       # ~ $33
        "Fuel": (3.8, 0.4),          # ~ $45
        "Restaurant": (4.0, 0.7),    # ~ $55
        "Online": (4.5, 1.0),        # ~ $90
        "Healthcare": (4.8, 0.8),    # ~ $120
        "Travel": (5.5, 0.9),        # ~ $245
        "Electronics": (5.8, 1.1),   # ~ $330
    }

    for i in range(num_rows):
        trans_id = start_id + i
        
        # Advance timestamp by Poisson-spaced minutes (e.g. 5 to 25 mins)
        time_delta_sec = int(rng.exponential(scale=600) + 120)
        current_dt += datetime.timedelta(seconds=time_delta_sec)
        timestamp_str = current_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        card_token = rng.choice(card_pool)
        merchant_id = rng.choice(merchant_pool)
        category = rng.choice(VALID_CATEGORIES, p=[0.12, 0.16, 0.14, 0.22, 0.18, 0.08, 0.10])
        card_type = rng.choice(VALID_CARD_TYPES, p=[0.60, 0.32, 0.08])
        
        # Base amount
        mu, sigma = cat_params[category]
        amount = float(np.clip(rng.lognormal(mu, sigma), 3.0, 5000.0))
        
        hour = current_dt.hour
        is_night = (hour >= 23 or hour < 5)
        
        # Determine fraud scenario
        # Target ~3% to 4% fraud rate
        # Scenario 1: Late night high-value online or electronics purchase (weight ~ 0.015)
        # Scenario 2: Sudden extreme luxury amount in Travel or Electronics (weight ~ 0.012)
        # Scenario 3: Medium-value repeated or off-pattern transaction (weight ~ 0.008)
        
        is_fraud = 0
        scenario_dice = rng.random()
        
        if is_night and category in ["Online", "Electronics", "Travel"] and amount > 400.0 and scenario_dice < 0.35:
            is_fraud = 1
        elif amount > 1800.0 and scenario_dice < 0.25:
            is_fraud = 1
        elif category in ["Electronics", "Online"] and amount > 750.0 and scenario_dice < 0.08:
            is_fraud = 1
        elif scenario_dice < 0.005:  # small random fraud across typical values (overlap)
            is_fraud = 1
            
        records.append({
            "trans_id": trans_id,
            "timestamp": timestamp_str,
            "card_token": card_token,
            "amount": round(amount, 2),
            "merchant_category": category,
            "is_fraud": is_fraud,
            "merchant_id": merchant_id,
            "card_type": card_type,
        })

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    fraud_count = int(df["is_fraud"].sum())
    manifest = {
        "manifest_version": "1.0",
        "description": "Synthetic demonstration transactions generated strictly per prompts/SYNTHETIC_DATA_GENERATION_PROMPT.txt.",
        "generation_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "random_seed": seed,
        "row_count": len(df),
        "column_names": list(df.columns),
        "trans_id_range": [int(df["trans_id"].min()), int(df["trans_id"].max())],
        "timestamp_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "fraud_count": fraud_count,
        "legitimate_count": len(df) - fraud_count,
        "fraud_rate": round(fraud_count / len(df), 4),
        "amount_stats": {
            "min": float(df["amount"].min()),
            "max": float(df["amount"].max()),
            "mean": round(float(df["amount"].mean()), 2),
            "median": round(float(df["amount"].median()), 2),
        },
        "cautionary_note": "This dataset is artificial and strictly isolated from dataset.csv. Never use synthetic data for production validation or claim real-world efficacy from it."
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=4500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output-csv", default="data/synthetic_transactions_only.csv", type=Path)
    parser.add_argument("--output-manifest", default="data/synthetic_data_manifest.json", type=Path)
    args = parser.parse_args()

    df = generate_synthetic_data(
        num_rows=args.rows,
        seed=args.seed,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
    )
    print(f"Generated {len(df)} synthetic rows saved to {args.output_csv}")
    print(f"Manifest written to {args.output_manifest}")
    print("Label counts:", df["is_fraud"].value_counts().to_dict())


if __name__ == "__main__":
    main()
