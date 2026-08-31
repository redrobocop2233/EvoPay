"""B12 -- Concept drift monitoring.

Tracks rolling feature distributions, fraud rate, model confidence,
and FPR/FNR over time windows. Uses PSI and KS-test for drift detection.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("evo-pay.drift")


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1: no significant shift
    PSI 0.1-0.2: moderate shift
    PSI > 0.2: significant shift

    Args:
        expected: Baseline distribution values.
        actual: Current distribution values.
        bins: Number of bins.

    Returns:
        PSI value (float).
    """
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # handle duplicates

    # Compute bin proportions
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Avoid division by zero
    expected_pct = (expected_counts + 1) / (expected_counts.sum() + len(expected_counts))
    actual_pct = (actual_counts + 1) / (actual_counts.sum() + len(actual_counts))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 6)


def compute_ks_stat(
    expected: np.ndarray,
    actual: np.ndarray,
) -> dict:
    """Kolmogorov-Smirnov test between two distributions.

    Returns:
        dict with ks_statistic and p_value
    """
    stat, p_value = stats.ks_2samp(expected, actual)
    return {
        "ks_statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "significant_drift": p_value < 0.05,
    }


def monitor_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
) -> dict:
    """Compute drift metrics for each feature.

    Args:
        baseline_df: Historical baseline data.
        current_df: Current/recent data.
        features: Feature columns to monitor.

    Returns:
        dict mapping feature_name -> {psi, ks_statistic, p_value, drift_level}
    """
    results = {}

    for feat in features:
        if feat not in baseline_df.columns or feat not in current_df.columns:
            continue

        baseline_vals = baseline_df[feat].dropna().values
        current_vals = current_df[feat].dropna().values

        if len(baseline_vals) < 10 or len(current_vals) < 10:
            continue

        psi = compute_psi(baseline_vals, current_vals)
        ks = compute_ks_stat(baseline_vals, current_vals)

        if psi > 0.2:
            drift_level = "HIGH"
        elif psi > 0.1:
            drift_level = "MODERATE"
        else:
            drift_level = "LOW"

        results[feat] = {
            "psi": psi,
            "ks_statistic": ks["ks_statistic"],
            "p_value": ks["p_value"],
            "drift_level": drift_level,
        }

    return results


def monitor_model_drift(
    baseline_scores: np.ndarray,
    baseline_labels: np.ndarray,
    current_scores: np.ndarray,
    current_labels: np.ndarray,
) -> dict:
    """Monitor drift in model outputs and performance.

    Returns:
        dict with score_distribution_drift, fraud_rate_change,
        performance_drift metrics.
    """
    # Score distribution drift
    score_psi = compute_psi(baseline_scores, current_scores)

    # Fraud rate change
    baseline_fraud_rate = baseline_labels.mean() if len(baseline_labels) > 0 else 0
    current_fraud_rate = current_labels.mean() if len(current_labels) > 0 else 0

    # Confidence drift (mean score)
    baseline_mean_score = baseline_scores.mean()
    current_mean_score = current_scores.mean()

    return {
        "score_psi": score_psi,
        "score_drift_level": "HIGH" if score_psi > 0.2 else ("MODERATE" if score_psi > 0.1 else "LOW"),
        "baseline_fraud_rate": round(float(baseline_fraud_rate), 4),
        "current_fraud_rate": round(float(current_fraud_rate), 4),
        "fraud_rate_change": round(float(current_fraud_rate - baseline_fraud_rate), 4),
        "baseline_mean_score": round(float(baseline_mean_score), 4),
        "current_mean_score": round(float(current_mean_score), 4),
    }
