"""B9 -- Score calibration (isotonic regression / Platt scaling).

Turns raw model scores into well-calibrated probabilities.
'A 0.7 should mean 70% likely fraud.'
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger("evo-pay.calibration")

MODEL_DIR = Path(__file__).resolve().parent / "saved"


def fit_calibrator(
    y_scores: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
) -> dict:
    """Fit a calibration model on held-out scores.

    Args:
        y_scores: Raw (uncalibrated) model scores.
        y_true: True binary labels.
        method: 'isotonic' or 'platt' (sigmoid).

    Returns:
        dict with: calibrator, method, metrics
    """
    if method == "isotonic":
        calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        calibrator.fit(y_scores, y_true)
    elif method == "platt":
        from sklearn.linear_model import LogisticRegression
        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(y_scores.reshape(-1, 1), y_true)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute calibration metrics
    calibrated = calibrate_batch({"calibrator": calibrator, "method": method}, y_scores)
    metrics = compute_calibration_metrics(y_true, calibrated)

    logger.info(
        "Calibrator fitted (%s): Brier=%.4f, ECE=%.4f",
        method, metrics["brier_score"], metrics["ece"],
    )

    return {
        "calibrator": calibrator,
        "method": method,
        "metrics": metrics,
    }


def calibrate(bundle: dict, raw_score: float) -> float:
    """Calibrate a single raw score to a true probability."""
    calibrator = bundle["calibrator"]
    method = bundle["method"]

    if method == "isotonic":
        result = calibrator.predict([raw_score])[0]
    elif method == "platt":
        result = calibrator.predict_proba([[raw_score]])[0, 1]
    else:
        result = raw_score

    return float(np.clip(result, 0.0, 1.0))


def calibrate_batch(bundle: dict, raw_scores: np.ndarray) -> np.ndarray:
    """Calibrate an array of raw scores."""
    calibrator = bundle["calibrator"]
    method = bundle["method"]

    if method == "isotonic":
        return np.clip(calibrator.predict(raw_scores), 0.0, 1.0)
    elif method == "platt":
        return calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    return raw_scores


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_calibrated: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute Brier score and Expected Calibration Error (ECE)."""
    from sklearn.metrics import brier_score_loss

    brier = brier_score_loss(y_true, y_calibrated)

    # ECE: weighted average of |accuracy - confidence| per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_calibrated >= bin_edges[i]) & (y_calibrated < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_calibrated[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return {
        "brier_score": round(float(brier), 6),
        "ece": round(float(ece), 6),
    }


def save_calibrator(bundle: dict, name: str = "calibrator") -> Path:
    """Save calibrator to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Calibrator saved to %s", path)
    return path


def load_calibrator(name: str = "calibrator") -> dict:
    """Load calibrator from disk."""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
