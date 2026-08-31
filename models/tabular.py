"""B3 — Supervised tabular fraud detection model.

Supports LightGBM (primary), RandomForest, and LogisticRegression.
Handles class imbalance via scale_pos_weight / class_weight.
Provides train, predict, save, and load interfaces.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("evo-pay.tabular")

# Default model save directory
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


def _get_pos_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight for imbalanced binary classification."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train_tabular_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "lightgbm",
) -> dict:
    """Train a tabular fraud detection model.

    Args:
        X_train, y_train: Training features and labels.
        X_val, y_val: Validation features and labels.
        model_type: 'lightgbm', 'random_forest', or 'logistic_regression'.

    Returns:
        dict with keys: model, model_type, feature_names, val_metrics, train_metrics
    """
    feature_names = list(X_train.columns)
    pos_weight = _get_pos_weight(y_train)
    logger.info(
        "Training %s — %d train / %d val — pos_weight=%.1f",
        model_type, len(X_train), len(X_val), pos_weight,
    )

    if model_type == "lightgbm":
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "scale_pos_weight": pos_weight,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 7,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    elif model_type == "logistic_regression":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )
        model.fit(X_train, y_train)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Evaluate ---
    val_scores = predict_tabular(model, X_val, model_type)
    train_scores = predict_tabular(model, X_train, model_type)

    val_metrics = _compute_metrics(y_val, val_scores)
    train_metrics = _compute_metrics(y_train, train_scores)

    logger.info("Val PR-AUC=%.4f  ROC-AUC=%.4f", val_metrics["pr_auc"], val_metrics["roc_auc"])

    return {
        "model": model,
        "model_type": model_type,
        "feature_names": feature_names,
        "val_metrics": val_metrics,
        "train_metrics": train_metrics,
    }


def predict_tabular(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "lightgbm",
) -> np.ndarray:
    """Return fraud probability scores from the tabular model.

    Args:
        model: Trained model object.
        X: Feature DataFrame.
        model_type: Type of model for correct predict call.

    Returns:
        1D numpy array of fraud probabilities.
    """
    if model_type == "lightgbm":
        return model.predict(X, num_iteration=model.best_iteration)
    else:
        return model.predict_proba(X)[:, 1]


def _compute_metrics(y_true: pd.Series, y_scores: np.ndarray) -> dict:
    """Compute key metrics for model evaluation."""
    from eval.metrics import compute_classification_metrics, recall_at_fpr

    y_pred = (y_scores >= 0.5).astype(int)
    metrics = compute_classification_metrics(
        y_true.values, y_pred, y_scores,
    )
    metrics["recall_at_1pct_fpr"] = recall_at_fpr(y_true.values, y_scores, 0.01)
    metrics["recall_at_5pct_fpr"] = recall_at_fpr(y_true.values, y_scores, 0.05)
    return metrics


def save_model(result: dict, name: str = "tabular_model") -> Path:
    """Save trained model and metadata to disk.

    Args:
        result: Output of train_tabular_model().
        name: Base filename (without extension).

    Returns:
        Path to the saved model file.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{name}.pkl"
    meta_path = MODEL_DIR / f"{name}_meta.json"

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(result["model"], f)

    # Save metadata (metrics + feature names)
    meta = {
        "model_type": result["model_type"],
        "feature_names": result["feature_names"],
        "val_metrics": {k: float(v) for k, v in result["val_metrics"].items()},
        "train_metrics": {k: float(v) for k, v in result["train_metrics"].items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model saved to %s", model_path)
    return model_path


def load_model(name: str = "tabular_model") -> dict:
    """Load a trained model and its metadata from disk.

    Returns:
        dict with keys: model, model_type, feature_names, val_metrics
    """
    model_path = MODEL_DIR / f"{name}.pkl"
    meta_path = MODEL_DIR / f"{name}_meta.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    return {
        "model": model,
        "model_type": meta["model_type"],
        "feature_names": meta["feature_names"],
        "val_metrics": meta.get("val_metrics", {}),
    }
