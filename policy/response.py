"""B8 -- Response policy: map calibrated risk score to actionable decision.

Cost-aware threshold optimization -- not just 'pick 0.5 arbitrarily'.
"""

import logging
from typing import Optional

import numpy as np
from api.schema import Decision

logger = logging.getLogger("evo-pay.policy")

DEFAULT_THRESHOLDS = {
    "allow_max": 0.20,      # below this -> ALLOW
    "challenge_max": 0.50,  # below this -> CHALLENGE
    "hold_max": 0.80,       # below this -> HOLD
    # above hold_max -> BLOCK
}

DEFAULT_COST_MATRIX = {
    "fraud_loss_per_dollar": 1.0,       # full dollar loss if fraud is missed
    "false_positive_friction": 0.05,    # cost per dollar of wrongly blocking legit
    "manual_review_cost": 2.50,         # fixed cost per HOLD review
    "challenge_cost": 0.50,             # friction cost of step-up auth
    "block_cost": 0.10,                 # customer anger / churn cost per block
}


def decide(
    risk_score: float,
    thresholds: Optional[dict] = None,
) -> Decision:
    """Map a calibrated risk score to an actionable decision.

    Args:
        risk_score: Calibrated fraud probability in [0, 1].
        thresholds: Override default thresholds.

    Returns:
        Decision enum: ALLOW, CHALLENGE, HOLD, or BLOCK.
    """
    t = thresholds or DEFAULT_THRESHOLDS

    if risk_score < t.get("allow_max", 0.20):
        return Decision.ALLOW
    elif risk_score < t.get("challenge_max", 0.50):
        return Decision.CHALLENGE
    elif risk_score < t.get("hold_max", 0.80):
        return Decision.HOLD
    else:
        return Decision.BLOCK


def expected_cost(
    risk_score: float,
    amount: float,
    decision: Decision,
    cost_matrix: Optional[dict] = None,
) -> float:
    """Compute expected cost of a decision given the risk score and amount.

    This is the objective we minimize when optimizing thresholds.
    """
    cm = cost_matrix or DEFAULT_COST_MATRIX
    p_fraud = risk_score

    if decision == Decision.ALLOW:
        # Risk: we miss the fraud
        return p_fraud * amount * cm["fraud_loss_per_dollar"]

    elif decision == Decision.CHALLENGE:
        # Partial friction + reduced fraud risk (challenge catches ~70% of fraud)
        return (
            p_fraud * amount * cm["fraud_loss_per_dollar"] * 0.3  # 30% still get through
            + cm["challenge_cost"]
        )

    elif decision == Decision.HOLD:
        # Manual review cost + very low fraud leakage (~5%)
        return (
            p_fraud * amount * cm["fraud_loss_per_dollar"] * 0.05
            + cm["manual_review_cost"]
        )

    elif decision == Decision.BLOCK:
        # No fraud loss, but friction cost if legitimate
        return (
            (1 - p_fraud) * cm["false_positive_friction"] * amount
            + cm["block_cost"]
        )

    return 0.0


def optimize_thresholds(
    y_scores: np.ndarray,
    y_true: np.ndarray,
    amounts: np.ndarray,
    cost_matrix: Optional[dict] = None,
    n_grid: int = 20,
) -> dict:
    """Find thresholds that minimize total expected cost via grid search.

    Args:
        y_scores: Calibrated risk scores.
        y_true: True fraud labels (0/1).
        amounts: Transaction amounts.
        cost_matrix: Cost parameters.
        n_grid: Grid resolution per threshold.

    Returns:
        dict with optimized thresholds and total cost.
    """
    cm = cost_matrix or DEFAULT_COST_MATRIX
    best_cost = float("inf")
    best_thresholds = DEFAULT_THRESHOLDS.copy()

    grid = np.linspace(0.05, 0.95, n_grid)

    for t1 in grid:
        for t2 in grid:
            if t2 <= t1:
                continue
            for t3 in grid:
                if t3 <= t2:
                    continue

                thresholds = {
                    "allow_max": t1,
                    "challenge_max": t2,
                    "hold_max": t3,
                }

                total_cost = 0.0
                for score, true_label, amount in zip(y_scores, y_true, amounts):
                    d = decide(score, thresholds)
                    total_cost += expected_cost(score, amount, d, cm)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_thresholds = thresholds.copy()

    logger.info(
        "Optimized thresholds: allow<%.2f, challenge<%.2f, hold<%.2f (cost=%.2f)",
        best_thresholds["allow_max"],
        best_thresholds["challenge_max"],
        best_thresholds["hold_max"],
        best_cost,
    )

    return {
        "thresholds": best_thresholds,
        "total_cost": best_cost,
    }
