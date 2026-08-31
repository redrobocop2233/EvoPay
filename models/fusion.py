"""B6 -- Risk score fusion across model components.

Combines tabular, anomaly, and graph scores via a learned logistic
regression combiner (not hand-tuned weights).
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("evo-pay.fusion")

MODEL_DIR = Path(__file__).resolve().parent / "saved"


def train_fusion_model(
    component_scores: pd.DataFrame,
    y_true: pd.Series,
) -> dict:
    """Train a logistic regression fusion model over component scores.

    Args:
        component_scores: DataFrame with columns like
            'tabular', 'anomaly', 'graph', etc.
        y_true: True fraud labels.

    Returns:
        dict with: model, feature_names, weights
    """
    feature_names = list(component_scores.columns)
    X = component_scores.fillna(0).values

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X, y_true)

    weights = dict(zip(feature_names, model.coef_[0].round(4)))
    logger.info("Fusion model trained. Weights: %s", weights)

    return {
        "model": model,
        "feature_names": feature_names,
        "weights": weights,
    }


def fuse_scores(
    bundle: dict,
    component_scores: dict,
) -> float:
    """Fuse component scores into a single risk score.

    Args:
        bundle: Output of train_fusion_model().
        component_scores: dict mapping component name to score.

    Returns:
        Fused probability score in [0, 1].
    """
    feature_names = bundle["feature_names"]
    X = np.array([[component_scores.get(fn, 0.0) for fn in feature_names]])
    prob = bundle["model"].predict_proba(X)[0, 1]
    return float(np.clip(prob, 0.0, 1.0))


def simple_weighted_fusion(
    component_scores: dict,
    weights: Optional[dict] = None,
) -> float:
    """Simple weighted average fusion (no training needed).

    Fallback when there's not enough data to train a fusion model.
    """
    if weights is None:
        weights = {
            "tabular": 0.60,
            "anomaly": 0.25,
            "graph": 0.10,
            "temporal": 0.05,
        }

    total_weight = 0.0
    weighted_sum = 0.0
    for comp, score in component_scores.items():
        w = weights.get(comp, 0.0)
        weighted_sum += score * w
        total_weight += w

    if total_weight > 0:
        return min(max(weighted_sum / total_weight, 0.0), 1.0)
    return 0.0


def save_fusion_model(bundle: dict, name: str = "fusion_model") -> Path:
    """Save fusion model to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Fusion model saved to %s", path)
    return path


def load_fusion_model(name: str = "fusion_model") -> dict:
    """Load fusion model from disk."""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
