"""B4 -- Unsupervised anomaly detection (IsolationForest).

Provides a label-free anomaly signal that complements the supervised
tabular model. Its primary value is detecting novel attack families
that the supervised model has never seen in training.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("evo-pay.anomaly")

MODEL_DIR = Path(__file__).resolve().parent / "saved"


def train_anomaly_model(
    X_train: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict:
    """Train an IsolationForest anomaly detector.

    Args:
        X_train: Feature matrix (same B1+B2 features used for tabular model).
        contamination: Expected fraction of anomalies.
        n_estimators: Number of trees.

    Returns:
        dict with: model, scaler, feature_names
    """
    feature_names = list(X_train.columns)

    # Scale features for IsolationForest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.fillna(0))

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    logger.info(
        "IsolationForest trained: %d features, %d samples, contamination=%.2f",
        len(feature_names), len(X_train), contamination,
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }


def predict_anomaly(
    bundle: dict,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return anomaly scores scaled to [0, 1] where higher = more anomalous.

    IsolationForest's decision_function returns negative scores for
    anomalies, so we negate and rescale to [0, 1].
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    # Align features
    X_aligned = X.reindex(columns=feature_names, fill_value=0).fillna(0)
    X_scaled = scaler.transform(X_aligned)

    # Raw scores: negative = anomaly, positive = normal
    raw_scores = model.decision_function(X_scaled)

    # Convert to [0, 1]: negate, then min-max scale
    negated = -raw_scores
    s_min, s_max = negated.min(), negated.max()
    if s_max > s_min:
        scaled = (negated - s_min) / (s_max - s_min)
    else:
        scaled = np.zeros_like(negated)

    return scaled


def predict_anomaly_single(bundle: dict, features: dict) -> float:
    """Score a single transaction's anomaly level (0-1, higher = more anomalous)."""
    feature_names = bundle["feature_names"]
    row = {fn: features.get(fn, 0.0) for fn in feature_names}
    X = pd.DataFrame([row])[feature_names]

    model = bundle["model"]
    scaler = bundle["scaler"]
    X_scaled = scaler.transform(X.fillna(0))

    raw = model.decision_function(X_scaled)[0]
    # Use the model's offset to normalize: scores < 0 are anomalies
    # Map [-0.5, 0.5] roughly to [1.0, 0.0]
    score = max(0.0, min(1.0, 0.5 - raw))
    return round(score, 4)


def save_anomaly_model(bundle: dict, name: str = "anomaly_model") -> Path:
    """Save anomaly model bundle to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Anomaly model saved to %s", path)
    return path


def load_anomaly_model(name: str = "anomaly_model") -> dict:
    """Load anomaly model bundle from disk."""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
