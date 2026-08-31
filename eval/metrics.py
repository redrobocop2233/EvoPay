"""Shared metric computation functions."""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix,
)
from typing import Optional


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: Optional[np.ndarray] = None) -> dict:
    """Compute precision, recall, F1, FPR, FNR, and optionally AUC metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    if y_scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        metrics["pr_auc"] = average_precision_score(y_true, y_scores)
        metrics["brier_score"] = brier_score_loss(y_true, y_scores)
    return metrics


def recall_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, max_fpr: float = 0.01) -> float:
    """Compute recall at a fixed FPR budget."""
    from sklearn.metrics import roc_curve
    fprs, tprs, _ = roc_curve(y_true, y_scores)
    # Find the highest TPR where FPR <= max_fpr
    valid = fprs <= max_fpr
    if not valid.any():
        return 0.0
    return float(tprs[valid][-1])
