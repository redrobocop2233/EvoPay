"""B10 -- Evaluation harness for attack family holdout testing.

Splits data by attack family (not random rows) to test generalization
to genuinely novel attack patterns.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from eval.metrics import compute_classification_metrics, recall_at_fpr

logger = logging.getLogger("evo-pay.harness")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate_by_attack_family(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    attack_families: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate detection performance broken down by attack family.

    Args:
        y_true: True labels (0/1).
        y_scores: Model risk scores.
        attack_families: Attack family label per transaction.
        threshold: Score threshold for binary prediction.

    Returns:
        dict mapping family_name -> metrics dict.
    """
    results = {}
    y_pred = (y_scores >= threshold).astype(int)

    for family in attack_families.unique():
        mask = attack_families == family
        n = mask.sum()
        if n < 2:
            continue

        fam_true = y_true[mask]
        fam_pred = y_pred[mask]
        fam_scores = y_scores[mask]

        if family == "legitimate":
            # For legit: report FPR
            fp = fam_pred.sum()
            results[family] = {
                "count": int(n),
                "false_positive_rate": round(float(fp / n), 4) if n > 0 else 0.0,
                "approval_rate": round(float(1 - fp / n), 4) if n > 0 else 1.0,
            }
        else:
            # For fraud families: report recall, precision, mean score
            if fam_true.sum() > 0:
                recall = float(fam_pred[fam_true == 1].mean())
            else:
                recall = 0.0
            results[family] = {
                "count": int(n),
                "fraud_count": int(fam_true.sum()),
                "recall": round(recall, 4),
                "mean_score": round(float(fam_scores.mean()), 4),
                "max_score": round(float(fam_scores.max()), 4),
                "min_score": round(float(fam_scores.min()), 4),
            }

    return results


def evaluate_novel_attacks(
    train_fn: Callable,
    predict_fn: Callable,
    feature_df: pd.DataFrame,
    holdout_family: str,
    feature_cols: list[str],
) -> dict:
    """Train on all families except holdout, test on holdout.

    This measures true generalization to a never-seen attack pattern.

    Args:
        train_fn: Function(X_train, y_train) -> model
        predict_fn: Function(model, X_test) -> scores
        feature_df: Full feature DataFrame with 'attack_family' and 'is_fraud'.
        holdout_family: Name of attack family to hold out entirely.
        feature_cols: List of feature column names.

    Returns:
        dict with train_metrics, holdout_metrics, holdout_family.
    """
    # Split: everything except holdout family for training
    train_mask = feature_df["attack_family"] != holdout_family
    test_mask = feature_df["attack_family"].isin([holdout_family, "legitimate"])

    X_train = feature_df.loc[train_mask, feature_cols].fillna(0)
    y_train = feature_df.loc[train_mask, "is_fraud"].astype(int)
    X_test = feature_df.loc[test_mask, feature_cols].fillna(0)
    y_test = feature_df.loc[test_mask, "is_fraud"].astype(int)

    logger.info(
        "Novel attack eval: holdout=%s, train=%d, test=%d (fraud=%d)",
        holdout_family, len(X_train), len(X_test), y_test.sum(),
    )

    # Train
    model = train_fn(X_train, y_train)

    # Predict
    scores = predict_fn(model, X_test)
    preds = (scores >= 0.5).astype(int)

    # Metrics
    overall = compute_classification_metrics(y_test.values, preds, scores)
    overall["recall_at_1pct_fpr"] = recall_at_fpr(y_test.values, scores, 0.01)

    # Holdout-specific metrics
    holdout_mask = feature_df.loc[test_mask, "attack_family"] == holdout_family
    if holdout_mask.sum() > 0:
        ho_scores = scores[holdout_mask.values]
        ho_true = y_test[holdout_mask].values
        ho_preds = (ho_scores >= 0.5).astype(int)
        holdout_recall = float(ho_preds[ho_true == 1].mean()) if (ho_true == 1).any() else 0.0
        holdout_metrics = {
            "count": int(holdout_mask.sum()),
            "recall": round(holdout_recall, 4),
            "mean_score": round(float(ho_scores.mean()), 4),
        }
    else:
        holdout_metrics = {"count": 0, "recall": 0.0, "mean_score": 0.0}

    return {
        "holdout_family": holdout_family,
        "overall_metrics": {k: round(float(v), 4) for k, v in overall.items()},
        "holdout_metrics": holdout_metrics,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def run_full_evaluation(
    train_fn: Callable,
    predict_fn: Callable,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    save_results: bool = True,
) -> dict:
    """Run the complete evaluation suite.

    1. Overall metrics on full test set
    2. Per-attack-family breakdown
    3. Novel attack holdout for each family

    Returns comprehensive evaluation report.
    """
    attack_families = [
        f for f in feature_df["attack_family"].unique()
        if f != "legitimate"
    ]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_rows": len(feature_df),
        "fraud_rate": round(float(feature_df["is_fraud"].mean()), 4),
        "attack_families": attack_families,
        "novel_attack_results": {},
    }

    # Novel attack holdout for each family
    for family in attack_families:
        logger.info("Running novel attack eval for: %s", family)
        try:
            result = evaluate_novel_attacks(
                train_fn, predict_fn, feature_df, family, feature_cols,
            )
            report["novel_attack_results"][family] = result
        except Exception as e:
            logger.error("Failed holdout eval for %s: %s", family, e)
            report["novel_attack_results"][family] = {"error": str(e)}

    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "evaluation_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Evaluation report saved to %s", path)

    return report
