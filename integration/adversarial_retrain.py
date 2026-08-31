"""Optional Blue-Team adversarial retraining hook.

The Red Team writes synthetic attack transactions into a training feed. The
existing Blue training pipeline can consume the merged base+adversarial data.
This module never changes a detector's decision directly; retraining is an
explicit, auditable operation between generations.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd


def red_transactions_frame(transactions) -> pd.DataFrame:
    rows = []
    for tx in transactions:
        row = asdict(tx)
        row["is_fraud"] = 1
        row.pop("label", None)
        row.pop("strategy_id", None)
        rows.append(row)
    return pd.DataFrame(rows)


def build_adversarial_feed(base_path: str | Path, transactions, output_path: str | Path) -> Path:
    base = pd.read_csv(base_path, parse_dates=["timestamp"])
    red = red_transactions_frame(transactions)
    if red.empty:
        merged = base
    else:
        # Keep only columns understood by the standard training data format.
        common = [c for c in base.columns if c in red.columns]
        merged = pd.concat([base[common], red[common]], ignore_index=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    return output


def retrain_blue(base_path: str | Path, transactions, output_dir: str | Path) -> dict:
    """Retrain the existing supervised/anomaly pipeline on base + Red feed."""
    from models.train_pipeline import build_feature_matrix, run_training
    from models.anomaly import train_anomaly_model, save_anomaly_model

    output_dir = Path(output_dir)
    feed_path = build_adversarial_feed(
        base_path, transactions, output_dir / "adversarial_training_feed.csv"
    )
    df = pd.read_csv(feed_path, parse_dates=["timestamp"])
    feature_df = build_feature_matrix(df, temporal_sample=min(5000, len(df)))
    training = run_training(feature_df)

    meta_cols = ["transaction_id", "is_fraud", "attack_family", "customer_id"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    X_all = feature_df[feature_cols].fillna(0)
    anomaly = train_anomaly_model(X_all, contamination=0.05)
    save_anomaly_model(anomaly)

    report = {
        "feed_path": str(feed_path),
        "rows": int(len(df)),
        "red_rows": int(len(transactions)),
        "best_model": training["report"]["best_model"],
        "test_metrics": training["test_metrics"],
    }
    (output_dir / "latest_retrain.json").write_text(json.dumps(report, indent=2))
    return report


def reload_api(base_url: str, timeout: int = 10) -> bool:
    """Ask a compatible local Blue API to reload models from disk."""
    import requests
    try:
        r = requests.post(f"{base_url.rstrip('/')}/reload", timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False
