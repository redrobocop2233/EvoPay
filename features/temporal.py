"""B2 — Rolling temporal / velocity features.

Computes transaction velocity and aggregates over sliding time windows.
All windows look strictly backward in time to prevent leakage.

Two interfaces:
  - compute_temporal_features()  — single transaction (API / real-time)
  - compute_temporal_features_batch() — full DataFrame (training)
"""

import numpy as np
import pandas as pd
from typing import Optional

# Window definitions: name -> pandas Timedelta
WINDOWS = {
    "5min": pd.Timedelta(minutes=5),
    "30min": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "6h": pd.Timedelta(hours=6),
    "24h": pd.Timedelta(hours=24),
    "7d": pd.Timedelta(days=7),
}


def _agg_window(
    history: pd.DataFrame,
    timestamp: pd.Timestamp,
    delta: pd.Timedelta,
) -> dict:
    """Aggregate statistics for transactions in [timestamp - delta, timestamp).

    Args:
        history: DataFrame of prior transactions for one customer,
            must have columns: amount, merchant_id, merchant_category,
            device_id, location_lat, location_lon, timestamp.
        timestamp: The current transaction's timestamp (exclusive upper bound).
        delta: How far back to look.

    Returns:
        dict with tx_count, amount_sum, amount_mean, unique_merchants,
        unique_devices, unique_locations.
    """
    cutoff = timestamp - delta
    mask = (history["timestamp"] >= cutoff) & (history["timestamp"] < timestamp)
    window = history.loc[mask]

    if window.empty:
        return {
            "tx_count": 0,
            "amount_sum": 0.0,
            "amount_mean": 0.0,
            "unique_merchants": 0,
            "unique_devices": 0,
            "unique_locations": 0,
        }

    amounts = window["amount"]

    # Unique locations: round lat/lon to 2 decimals (~1 km grid) then count
    if "location_lat" in window.columns and "location_lon" in window.columns:
        loc_pairs = set(
            zip(
                window["location_lat"].round(2),
                window["location_lon"].round(2),
            )
        )
        unique_locations = len(loc_pairs)
    else:
        unique_locations = 0

    return {
        "tx_count": len(window),
        "amount_sum": round(float(amounts.sum()), 2),
        "amount_mean": round(float(amounts.mean()), 2),
        "unique_merchants": int(window["merchant_id"].nunique()) if "merchant_id" in window.columns else 0,
        "unique_devices": int(window["device_id"].nunique()) if "device_id" in window.columns else 0,
        "unique_locations": unique_locations,
    }


def compute_temporal_features(
    customer_id: str,
    timestamp,
    history_df: pd.DataFrame,
) -> dict:
    """Compute rolling window features for a single transaction.

    Args:
        customer_id: The customer to query.
        timestamp: The current transaction's timestamp (str or pd.Timestamp).
        history_df: Full transaction history DataFrame (will be filtered
            to this customer and timestamps before the current one).

    Returns:
        Flat dict with keys like tx_count_5min, amount_sum_30min, etc.
    """
    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.Timestamp(timestamp)

    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(history_df["timestamp"]):
        history_df = history_df.copy()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

    # Filter to this customer, strictly before current timestamp
    cust_hist = history_df[
        (history_df["customer_id"] == customer_id)
        & (history_df["timestamp"] < timestamp)
    ]

    features: dict = {}
    for win_name, delta in WINDOWS.items():
        agg = _agg_window(cust_hist, timestamp, delta)
        for metric, value in agg.items():
            features[f"{metric}_{win_name}"] = value

    # --- Derived velocity signals ---
    # Acceleration: ratio of short-window count to longer window count
    c5 = features.get("tx_count_5min", 0)
    c1h = features.get("tx_count_1h", 0)
    c24h = features.get("tx_count_24h", 0)

    features["velocity_ratio_5min_1h"] = round(c5 / max(c1h, 1), 4)
    features["velocity_ratio_1h_24h"] = round(c1h / max(c24h, 1), 4)

    # Amount burst: ratio of 5min sum to 24h mean (per-tx)
    s5 = features.get("amount_sum_5min", 0)
    m24 = features.get("amount_mean_24h", 0)
    features["amount_burst_ratio"] = round(s5 / max(m24, 1), 4)

    return features


def compute_temporal_features_batch(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Vectorized temporal feature computation for training data.

    Iterates through sorted transactions per customer and computes
    rolling window features using only backward-looking history
    to prevent data leakage.

    Args:
        df: Full transaction DataFrame with columns: customer_id,
            amount, merchant_id, merchant_category, device_id,
            location_lat, location_lon, timestamp.
        sample_size: If set, only compute features for a random
            sample of rows (for speed during development).

    Returns:
        DataFrame with transaction_id as index/column and all
        temporal feature columns.
    """
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by time globally
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    if sample_size and sample_size < len(df_sorted):
        # Deterministic sample — pick indices spread across the timeline
        rng = np.random.RandomState(42)
        sample_idx = sorted(rng.choice(len(df_sorted), sample_size, replace=False))
        targets = df_sorted.iloc[sample_idx]
    else:
        targets = df_sorted

    rows = []
    # Pre-group history by customer for O(1) lookup
    grouped = {cid: grp.sort_values("timestamp") for cid, grp in df_sorted.groupby("customer_id")}

    for _, row in targets.iterrows():
        cid = row["customer_id"]
        ts = row["timestamp"]
        cust_hist = grouped.get(cid, pd.DataFrame())

        # Only look at transactions strictly before this one
        prior = cust_hist[cust_hist["timestamp"] < ts]

        features: dict = {"transaction_id": row.get("transaction_id")}

        for win_name, delta in WINDOWS.items():
            agg = _agg_window(prior, ts, delta)
            for metric, value in agg.items():
                features[f"{metric}_{win_name}"] = value

        # Derived velocity signals
        c5 = features.get("tx_count_5min", 0)
        c1h = features.get("tx_count_1h", 0)
        c24h = features.get("tx_count_24h", 0)
        s5 = features.get("amount_sum_5min", 0)
        m24 = features.get("amount_mean_24h", 0)

        features["velocity_ratio_5min_1h"] = round(c5 / max(c1h, 1), 4)
        features["velocity_ratio_1h_24h"] = round(c1h / max(c24h, 1), 4)
        features["amount_burst_ratio"] = round(s5 / max(m24, 1), 4)

        rows.append(features)

    return pd.DataFrame(rows)
