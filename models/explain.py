"""B7 -- SHAP-based explainability and reason code generation.

Uses SHAP TreeExplainer (for LightGBM/RF) or KernelExplainer (for LR)
to derive per-transaction reason codes backed by actual model evidence.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

# imported at module level (not lazily inside the request path) so the one-time
# import/JIT warmup cost (roughly 0.7-0.9s observed) is paid once when the
# server process starts, not on whichever live request happens to be first
import shap

logger = logging.getLogger("evo-pay.explain")

# TreeExplainer construction walks the whole booster's tree structure -
# cheap per call in isolation (~10-30ms observed here) but wasted work when
# it's the same model every time. Cached by id(model) so a reloaded/retrained
# model (a new object) doesn't reuse a stale explainer.
_EXPLAINER_CACHE: dict[int, Any] = {}


def _get_tree_explainer(model: Any) -> Any:
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE.clear()  # only one model is ever loaded at a time in this API
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[key]

# Map raw feature names to human-readable reason codes
REASON_CODE_MAP = {
    "amount_zscore": "customer_amount_deviation",
    "is_new_device": "new_device",
    "is_new_merchant_category": "new_merchant",
    "hour_deviation": "unusual_time",
    "location_distance_km": "unusual_location",
    "is_thin_file": "thin_customer_history",
    "tx_count_5min": "abnormal_velocity",
    "tx_count_30min": "high_frequency_burst",
    "tx_count_1h": "high_frequency_burst",
    "amount_sum_5min": "amount_burst",
    "unique_merchants_5min": "merchant_spray",
    "unique_merchants_30min": "merchant_spray",
    "unique_merchants_1h": "merchant_spray",
    "unique_devices_1h": "device_switching",
    "velocity_ratio_5min_1h": "velocity_acceleration",
    "velocity_ratio_1h_24h": "velocity_acceleration",
    "amount_burst_ratio": "amount_burst",
    "shared_device_count": "graph_anomaly",
    "device_customer_degree": "graph_anomaly",
    "merchant_fraud_rate": "graph_anomaly",
    "customer_degree": "graph_anomaly",
    "is_part_of_dense_cluster": "graph_anomaly",
}

# Fallback for features not in the map: strip suffixes and try again
_WINDOW_SUFFIXES = ["_5min", "_30min", "_1h", "_6h", "_24h", "_7d"]


def _feature_to_reason_code(feature_name: str) -> str:
    """Convert a feature name to a human-readable reason code."""
    if feature_name in REASON_CODE_MAP:
        return REASON_CODE_MAP[feature_name]

    # Try stripping window suffixes
    for suffix in _WINDOW_SUFFIXES:
        base = feature_name.replace(suffix, "")
        if base in REASON_CODE_MAP:
            return REASON_CODE_MAP[base]

    # Fallback: use the feature name itself
    return feature_name


def generate_reason_codes(
    model: Any,
    features: pd.DataFrame,
    model_type: str = "lightgbm",
    top_n: int = 3,
) -> list[str]:
    """Generate human-readable reason codes using SHAP values.

    Args:
        model: Trained model object.
        features: Single-row DataFrame of features.
        model_type: Type of model for SHAP explainer selection.
        top_n: Number of top contributing features to include.

    Returns:
        List of human-readable reason codes, deduplicated.
    """
    try:
        if model_type in ("lightgbm", "random_forest"):
            explainer = _get_tree_explainer(model)
            shap_values = explainer.shap_values(features)
            # For binary classification, TreeExplainer may return a list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # class 1 (fraud)
        else:
            # For logistic regression, use a simpler approach
            # Get coefficients as pseudo-SHAP values
            shap_values = _coefficient_importance(model, features)

    except Exception as e:
        logger.warning("SHAP computation failed: %s -- falling back to coefficient method", e)
        shap_values = _coefficient_importance(model, features)

    # Get top contributing features (by absolute SHAP value)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
    else:
        shap_values = np.array(shap_values).flatten()

    feature_names = list(features.columns)
    abs_shap = np.abs(shap_values)

    # Only include features with positive contribution to fraud
    top_indices = np.argsort(abs_shap)[::-1][:top_n * 2]  # grab extra, filter positive

    codes = []
    seen = set()
    for idx in top_indices:
        if len(codes) >= top_n:
            break
        if shap_values[idx] > 0:  # positive = pushes toward fraud
            code = _feature_to_reason_code(feature_names[idx])
            if code not in seen:
                codes.append(code)
                seen.add(code)

    return codes if codes else ["model_prediction"]


def _coefficient_importance(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Use model coefficients as pseudo-SHAP importance for linear models."""
    try:
        coefs = model.coef_[0]  # shape (n_features,)
        values = features.values[0]  # shape (n_features,)
        return coefs * values  # element-wise contribution
    except Exception:
        return np.zeros(features.shape[1])


def generate_reason_codes_from_features(
    behavioral: dict,
    temporal: dict,
    threshold_overrides: Optional[dict] = None,
) -> list[str]:
    """Rule-based reason codes as fallback when SHAP is unavailable.

    Each code is still backed by actual feature evidence for the
    specific transaction.
    """
    codes: list[str] = []

    # Behavioral (B1)
    if abs(behavioral.get("amount_zscore", 0)) > 2:
        codes.append("customer_amount_deviation")
    if behavioral.get("is_new_device", 0):
        codes.append("new_device")
    if behavioral.get("is_new_merchant_category", 0):
        codes.append("new_merchant")
    if behavioral.get("hour_deviation", 0) >= 3:
        codes.append("unusual_time")
    if behavioral.get("location_distance_km", 0) > 100:
        codes.append("unusual_location")
    if behavioral.get("is_thin_file", 0):
        codes.append("thin_customer_history")

    # Temporal (B2)
    if temporal.get("tx_count_5min", 0) >= 3:
        codes.append("abnormal_velocity")
    if temporal.get("tx_count_30min", 0) >= 10:
        codes.append("high_frequency_burst")
    if temporal.get("unique_merchants_30min", 0) >= 5:
        codes.append("merchant_spray")
    if temporal.get("velocity_ratio_5min_1h", 0) >= 0.8:
        codes.append("velocity_acceleration")

    return codes
