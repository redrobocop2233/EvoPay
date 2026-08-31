"""B1 — Customer behavioral baseline features.

Computes per-customer deviation features by comparing a transaction
against the customer's historical behavioral profile.

Profiles are built from historical data and store distributional
summaries (mean, std, sets of known values, centroid location).
At scoring time, each transaction is compared against its customer's
profile to produce deviation signals — these are the features that
downstream models consume, NOT raw transaction fields.
"""

import math
import numpy as np
import pandas as pd
from typing import Optional


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * math.asin(math.sqrt(a))


def _circular_hour_distance(h1: float, h2: float) -> float:
    """Minimum angular distance between two hour-of-day values (0-24 scale).

    Returns a value in [0, 12] — e.g. hour 23 and hour 1 are 2 apart,
    not 22 apart.
    """
    diff = abs(h1 - h2) % 24
    return min(diff, 24 - diff)


def build_customer_profiles(history_df: pd.DataFrame) -> dict:
    """Build behavioral profiles for all customers from historical data.

    Args:
        history_df: DataFrame with columns: customer_id, amount,
            merchant_category, device_id, location_lat, location_lon,
            timestamp (datetime-like).

    Returns:
        dict mapping customer_id -> profile dict with keys:
            amount_mean, amount_std, typical_merchants (set),
            typical_hours (list of ints), typical_devices (set),
            home_lat, home_lon, tx_count
    """
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(history_df["timestamp"]):
        history_df = history_df.copy()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])

    profiles: dict = {}

    for cid, grp in history_df.groupby("customer_id"):
        amounts = grp["amount"]
        hours = grp["timestamp"].dt.hour

        profiles[cid] = {
            # Amount distribution
            "amount_mean": float(amounts.mean()),
            "amount_std": float(amounts.std()) if len(amounts) > 1 else float(amounts.mean() * 0.3),
            # Known merchant categories
            "typical_merchants": set(grp["merchant_category"].dropna().unique()),
            # Typical transacting hours (as list of ints)
            "typical_hours": sorted(hours.unique().tolist()),
            # Known devices
            "typical_devices": set(grp["device_id"].dropna().unique()),
            # Home location — centroid of historical transactions
            "home_lat": float(grp["location_lat"].mean()),
            "home_lon": float(grp["location_lon"].mean()),
            # Volume (useful for thin-file detection)
            "tx_count": len(grp),
        }

    return profiles


def compute_behavioral_features(transaction: dict, profile: Optional[dict]) -> dict:
    """Compute deviation features for a single transaction vs its customer profile.

    If no profile exists (new customer / thin file), returns elevated-risk
    defaults — absence of history is itself a signal.

    Args:
        transaction: dict with keys amount, merchant_category, device_id,
            location_lat, location_lon, timestamp.
        profile: output of build_customer_profiles for this customer,
            or None if the customer has no history.

    Returns:
        dict with keys:
            amount_zscore          — how many σ above/below the customer mean
            is_new_merchant_category — 1 if merchant cat not in history, else 0
            is_new_device          — 1 if device_id not in history, else 0
            hour_deviation         — min circular distance to any typical hour
            location_distance_km   — km from customer's home centroid
            is_thin_file           — 1 if customer has < 5 historical txns
    """
    # ---- No-profile defaults (new / synthetic identity) ----
    if profile is None:
        return {
            "amount_zscore": 3.0,           # treat as highly unusual
            "is_new_merchant_category": 1,
            "is_new_device": 1,
            "hour_deviation": 12.0,         # max possible
            "location_distance_km": 10000.0,  # flag as distant
            "is_thin_file": 1,
        }

    # ---- Amount z-score ----
    amt = float(transaction.get("amount", 0))
    std = profile["amount_std"]
    if std > 0:
        amount_zscore = (amt - profile["amount_mean"]) / std
    else:
        amount_zscore = 0.0 if amt == profile["amount_mean"] else 3.0

    # ---- New merchant category ----
    merchant_cat = transaction.get("merchant_category", "")
    is_new_merchant = int(merchant_cat not in profile["typical_merchants"])

    # ---- New device ----
    device_id = transaction.get("device_id", "")
    is_new_device = int(device_id not in profile["typical_devices"])

    # ---- Hour deviation ----
    ts = transaction.get("timestamp")
    if ts is not None:
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        tx_hour = ts.hour if hasattr(ts, "hour") else 12
    else:
        tx_hour = 12  # default to noon if missing

    if profile["typical_hours"]:
        hour_deviation = min(
            _circular_hour_distance(tx_hour, h) for h in profile["typical_hours"]
        )
    else:
        hour_deviation = 0.0

    # ---- Location distance ----
    tx_lat = float(transaction.get("location_lat", profile["home_lat"]))
    tx_lon = float(transaction.get("location_lon", profile["home_lon"]))
    location_distance_km = _haversine_km(
        profile["home_lat"], profile["home_lon"], tx_lat, tx_lon
    )

    # ---- Thin file flag ----
    is_thin_file = int(profile["tx_count"] < 5)

    return {
        "amount_zscore": round(amount_zscore, 4),
        "is_new_merchant_category": is_new_merchant,
        "is_new_device": is_new_device,
        "hour_deviation": round(hour_deviation, 2),
        "location_distance_km": round(location_distance_km, 2),
        "is_thin_file": is_thin_file,
    }


def compute_behavioral_features_batch(
    df: pd.DataFrame,
    profiles: Optional[dict] = None,
) -> pd.DataFrame:
    """Vectorized behavioral feature computation for training data.

    If profiles is None, builds them from the legitimate transactions
    in df (attack_family == 'legitimate'), then scores every row.

    Args:
        df: Full transaction DataFrame.
        profiles: Pre-built profiles dict, or None to auto-build.

    Returns:
        DataFrame with one row per transaction and behavioral feature columns.
    """
    if profiles is None:
        legit = df[df["attack_family"] == "legitimate"] if "attack_family" in df.columns else df
        profiles = build_customer_profiles(legit)

    rows = []
    for _, row in df.iterrows():
        tx = row.to_dict()
        cid = tx.get("customer_id")
        profile = profiles.get(cid)
        feats = compute_behavioral_features(tx, profile)
        feats["transaction_id"] = tx.get("transaction_id")
        rows.append(feats)

    return pd.DataFrame(rows)
