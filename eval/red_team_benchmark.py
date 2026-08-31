"""
Defend pillar evaluation (per the challenge brief: "maximise detection
performance (precision, recall, F1/AUC) on the simulated attacks while
keeping false positives on legitimate payments low").

Every number reported so far in this project (detection_rate,
FailureAnalyzer's per-dimension breakdown) has been recall-only: fraction
of fraud campaigns Blue flagged. That's half of what the brief actually
asks for. This script runs the whole evaluation: Blue's detector against
BOTH a red-team-evolved fraud population AND Red's own legitimate
transactions, producing precision, recall, F1, ROC-AUC, and false positive
rate - and recall specifically at a fixed FPR budget, since a detector that
also flags 40% of legitimate payments is not a usable one regardless of
how much fraud it catches.

Usage (from evo-pay/, with the Blue API running):
    python -m eval.red_team_benchmark
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import compute_classification_metrics, recall_at_fpr


def run_benchmark(num_customers=150, num_merchants=40, num_days=60,
                   population_size=30, generations=10, api_url="http://127.0.0.1:8000",
                   num_legit_samples=400, seed=11):
    from red_and_blue_team.ecosystem import PaymentEcosystem
    from red_and_blue_team.blue_team_client import BlueTeamClient, wait_for_api
    from red_and_blue_team.blue_team import HeuristicDetector
    from red_and_blue_team.red_team import RedTeamController

    print("Checking Blue API...")
    wait_for_api(api_url)

    print(f"Building ecosystem ({num_customers} customers, {num_days} days)...")
    eco = PaymentEcosystem(num_customers=num_customers, num_merchants=num_merchants, num_days=num_days)
    eco.generate_transactions()

    print(f"Evolving red team ({population_size} population x {generations} generations)...")
    detector = BlueTeamClient(base_url=api_url, fallback=HeuristicDetector())
    controller = RedTeamController(eco, detector=detector, population_size=population_size)
    memory = controller.evolve(generations=generations)

    # --- fraud side: every evaluated campaign, scored by Blue ---
    fraud_scores = np.array([r.risk_score for r in memory.records])
    fraud_labels = np.ones(len(fraud_scores))
    print(f"Fraud campaigns evaluated: {len(fraud_scores)}")

    # --- legit side: real Red-generated legit transactions, scored by Blue ---
    # (not campaigns the red team touched at all - the actual false-positive test)
    import random as _random
    rng = _random.Random(seed)
    legit_sample = rng.sample(eco.transactions, min(num_legit_samples, len(eco.transactions)))
    print(f"Scoring {len(legit_sample)} legitimate (never-attacked) transactions...")

    legit_scores = []
    for txn in legit_sample:
        history = detector._history_payload(txn.customer_id, eco)
        try:
            result = detector._evaluate_one(txn, history)
            legit_scores.append(result["risk_score"])
        except Exception as e:
            print(f"  (skipped one legit transaction: {e})")
    legit_scores = np.array(legit_scores)
    legit_labels = np.zeros(len(legit_scores))

    # --- combine and compute the actual Defend-pillar metrics ---
    y_true = np.concatenate([fraud_labels, legit_labels])
    y_scores = np.concatenate([fraud_scores, legit_scores])
    y_pred = (y_scores >= 0.5).astype(int)

    metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    metrics["recall_at_1pct_fpr"] = recall_at_fpr(y_true, y_scores, max_fpr=0.01)
    metrics["recall_at_5pct_fpr"] = recall_at_fpr(y_true, y_scores, max_fpr=0.05)
    metrics["n_fraud"] = len(fraud_scores)
    metrics["n_legit"] = len(legit_scores)
    metrics["mean_fraud_score"] = round(float(fraud_scores.mean()), 4)
    metrics["mean_legit_score"] = round(float(legit_scores.mean()), 4) if len(legit_scores) else None

    return metrics, memory, controller


def print_report(metrics):
    print()
    print("=" * 60)
    print("DEFEND PILLAR EVALUATION")
    print("=" * 60)
    print(f"  Fraud campaigns:          {metrics['n_fraud']}")
    print(f"  Legit transactions:       {metrics['n_legit']}")
    print(f"  Mean fraud risk_score:    {metrics['mean_fraud_score']}")
    print(f"  Mean legit risk_score:    {metrics['mean_legit_score']}")
    print("-" * 60)
    print(f"  Precision:                {metrics['precision']:.4f}")
    print(f"  Recall:                   {metrics['recall']:.4f}")
    print(f"  F1:                       {metrics['f1']:.4f}")
    print(f"  False positive rate:      {metrics.get('fpr', 'n/a')}")
    print(f"  ROC-AUC:                  {metrics.get('roc_auc', 'n/a')}")
    print(f"  PR-AUC:                   {metrics.get('pr_auc', 'n/a')}")
    print(f"  Recall @ 1% FPR budget:   {metrics['recall_at_1pct_fpr']:.4f}")
    print(f"  Recall @ 5% FPR budget:   {metrics['recall_at_5pct_fpr']:.4f}")
    print("=" * 60)
    if metrics["mean_legit_score"] and metrics["mean_legit_score"] > 0.15:
        print("NOTE: mean legit score is meaningfully above 0 - see INTEGRATION.md")
        print("re: the Red/Blue synthetic-world calibration gap before trusting")
        print("precision/FPR numbers above at face value.")


if __name__ == "__main__":
    metrics, memory, controller = run_benchmark()
    print_report(metrics)
