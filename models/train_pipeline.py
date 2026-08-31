"""End-to-end training pipeline for B3 tabular model.

Usage:
    python -m models.train_pipeline

Steps:
  1. Load synthetic dataset
  2. Build behavioral features (B1)
  3. Build temporal features (B2) — sampled for speed
  4. Combine into feature matrix
  5. Split: train / val / test (stratified, time-aware)
  6. Train LightGBM, RandomForest, LogisticRegression
  7. Compare and select best model on PR-AUC
  8. Save best model to disk
  9. Print full evaluation report
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Setup path so imports work when run as module ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders import load_synthetic
from features.behavioral import build_customer_profiles, compute_behavioral_features_batch
from features.temporal import compute_temporal_features_batch
from models.tabular import train_tabular_model, save_model, predict_tabular
from eval.metrics import compute_classification_metrics, recall_at_fpr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("evo-pay.train")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"


def build_feature_matrix(df: pd.DataFrame, temporal_sample: int = 5000) -> pd.DataFrame:
    """Build combined B1 + B2 feature matrix from raw transactions.

    Args:
        df: Full transaction DataFrame.
        temporal_sample: Max rows for temporal feature computation
            (temporal is O(n*m) and slow on large data).

    Returns:
        DataFrame with transaction_id, all feature columns, is_fraud, attack_family.
    """
    logger.info("Building feature matrix from %d transactions...", len(df))

    # --- B1: Behavioral features ---
    t0 = time.time()
    legit = df[df["attack_family"] == "legitimate"] if "attack_family" in df.columns else df
    profiles = build_customer_profiles(legit)
    behavioral_df = compute_behavioral_features_batch(df, profiles)
    logger.info("B1 behavioral features: %.1fs", time.time() - t0)

    # --- B2: Temporal features (sampled for speed) ---
    t0 = time.time()
    temporal_df = compute_temporal_features_batch(df, sample_size=temporal_sample)
    logger.info("B2 temporal features: %.1fs (%d rows)", time.time() - t0, len(temporal_df))

    # --- Merge on transaction_id ---
    merged = behavioral_df.merge(temporal_df, on="transaction_id", how="inner")

    # Attach labels
    label_cols = ["transaction_id", "is_fraud", "attack_family", "customer_id"]
    labels = df[[c for c in label_cols if c in df.columns]]
    merged = merged.merge(labels, on="transaction_id", how="inner")

    logger.info("Feature matrix: %d rows x %d columns", len(merged), len(merged.columns))
    return merged


def run_training(feature_df: pd.DataFrame) -> dict:
    """Run the full training and evaluation pipeline.

    Returns:
        dict with best_result, all_results, and report.
    """
    # --- Separate features, labels, metadata ---
    meta_cols = ["transaction_id", "is_fraud", "attack_family", "customer_id"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    X = feature_df[feature_cols].copy()
    y = feature_df["is_fraud"].astype(int)

    # Fill NaN with 0 (missing temporal features for early transactions)
    X = X.fillna(0)

    logger.info("Features: %d | Fraud rate: %.2f%%", len(feature_cols), y.mean() * 100)

    # --- Stratified split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42,
    )

    logger.info(
        "Split: %d train / %d val / %d test (fraud: %.1f%% / %.1f%% / %.1f%%)",
        len(X_train), len(X_val), len(X_test),
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )

    # --- Train multiple models ---
    model_types = ["lightgbm", "random_forest", "logistic_regression"]
    all_results = {}

    for mt in model_types:
        logger.info("=" * 60)
        logger.info("Training: %s", mt)
        try:
            result = train_tabular_model(X_train, y_train, X_val, y_val, mt)
            all_results[mt] = result
        except Exception as e:
            logger.error("Failed to train %s: %s", mt, e)

    # --- Select best on validation PR-AUC ---
    best_name = max(
        all_results,
        key=lambda k: all_results[k]["val_metrics"].get("pr_auc", 0),
    )
    best_result = all_results[best_name]
    logger.info("Best model: %s (val PR-AUC=%.4f)", best_name, best_result["val_metrics"]["pr_auc"])

    # --- Evaluate best model on test set ---
    test_scores = predict_tabular(best_result["model"], X_test, best_result["model_type"])
    test_pred = (test_scores >= 0.5).astype(int)
    test_metrics = compute_classification_metrics(y_test.values, test_pred, test_scores)
    test_metrics["recall_at_1pct_fpr"] = recall_at_fpr(y_test.values, test_scores, 0.01)
    test_metrics["recall_at_5pct_fpr"] = recall_at_fpr(y_test.values, test_scores, 0.05)

    # --- Per-attack-family breakdown (if available) ---
    attack_breakdown = {}
    if "attack_family" in feature_df.columns:
        test_families = feature_df.loc[X_test.index, "attack_family"]
        for family in test_families.unique():
            if family == "legitimate":
                continue
            mask = test_families == family
            if mask.sum() < 2:
                continue
            fam_scores = test_scores[mask.values]
            fam_true = y_test[mask].values
            fam_pred = (fam_scores >= 0.5).astype(int)
            attack_breakdown[family] = {
                "count": int(mask.sum()),
                "recall": float(fam_pred[fam_true == 1].mean()) if (fam_true == 1).any() else 0.0,
                "mean_score": float(fam_scores.mean()),
            }

    # --- Build report ---
    report = {
        "best_model": best_name,
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "data_split": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "validation_metrics": {
            mt: {k: round(float(v), 4) for k, v in r["val_metrics"].items()}
            for mt, r in all_results.items()
        },
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "attack_family_breakdown": attack_breakdown,
    }

    # --- Save best model ---
    save_model(best_result, "tabular_model")

    # --- Save report ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)

    return {
        "best_result": best_result,
        "all_results": all_results,
        "report": report,
        "test_metrics": test_metrics,
    }


def print_report(report: dict) -> None:
    """Pretty-print the training report."""
    print("\n" + "=" * 70)
    print("  EVO-PAY BLUE TEAM -- TABULAR MODEL TRAINING REPORT")
    print("=" * 70)

    print(f"\n Best Model: {report['best_model']}")
    print(f" Features: {report['n_features']}")
    print(f" Data: {report['data_split']}")

    print("\n" + "-" * 70)
    print("  VALIDATION METRICS (model comparison)")
    print("-" * 70)
    header = f"{'Model':<22} {'PR-AUC':>8} {'ROC-AUC':>8} {'Recall':>8} {'Prec':>8} {'F1':>8} {'R@1%FPR':>8}"
    print(header)
    for mt, metrics in report["validation_metrics"].items():
        marker = " [*]" if mt == report["best_model"] else ""
        print(
            f"{mt + marker:<22} {metrics.get('pr_auc', 0):>8.4f} "
            f"{metrics.get('roc_auc', 0):>8.4f} {metrics.get('recall', 0):>8.4f} "
            f"{metrics.get('precision', 0):>8.4f} {metrics.get('f1', 0):>8.4f} "
            f"{metrics.get('recall_at_1pct_fpr', 0):>8.4f}"
        )

    print("\n" + "-" * 70)
    print("  TEST SET METRICS (best model)")
    print("-" * 70)
    tm = report["test_metrics"]
    print(f"  PR-AUC:          {tm.get('pr_auc', 0):.4f}")
    print(f"  ROC-AUC:         {tm.get('roc_auc', 0):.4f}")
    print(f"  Precision:       {tm.get('precision', 0):.4f}")
    print(f"  Recall:          {tm.get('recall', 0):.4f}")
    print(f"  F1:              {tm.get('f1', 0):.4f}")
    print(f"  FPR:             {tm.get('fpr', 0):.4f}")
    print(f"  FNR:             {tm.get('fnr', 0):.4f}")
    print(f"  Recall @ 1% FPR: {tm.get('recall_at_1pct_fpr', 0):.4f}")
    print(f"  Recall @ 5% FPR: {tm.get('recall_at_5pct_fpr', 0):.4f}")
    print(f"  Brier Score:     {tm.get('brier_score', 0):.4f}")

    if report.get("attack_family_breakdown"):
        print("\n" + "-" * 70)
        print("  PER-ATTACK-FAMILY RECALL (test set)")
        print("-" * 70)
        for family, stats in sorted(report["attack_family_breakdown"].items()):
            print(f"  {family:<25} recall={stats['recall']:.3f}  mean_score={stats['mean_score']:.3f}  n={stats['count']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logger.info("Starting training pipeline...")
    t_start = time.time()

    # 1. Load data
    df = load_synthetic()
    logger.info("Loaded %d transactions", len(df))

    # 2. Build feature matrix
    feature_df = build_feature_matrix(df, temporal_sample=5000)

    # 3. Train tabular models and evaluate
    output = run_training(feature_df)

    # 4. Train anomaly model (unsupervised)
    from models.anomaly import train_anomaly_model, save_anomaly_model
    meta_cols = ["transaction_id", "is_fraud", "attack_family", "customer_id"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    X_all = feature_df[feature_cols].fillna(0)
    anomaly_bundle = train_anomaly_model(X_all, contamination=0.05)
    save_anomaly_model(anomaly_bundle)
    logger.info("Anomaly model trained and saved")

    # 5. Print report
    print_report(output["report"])

    elapsed = time.time() - t_start
    logger.info("Total pipeline time: %.1fs", elapsed)

    elapsed = time.time() - t_start
    logger.info("Total pipeline time: %.1fs", elapsed)
