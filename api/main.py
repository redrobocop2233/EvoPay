"""FastAPI application entrypoint for EVO-PAY Blue Team fraud detection.

The /evaluate endpoint runs the full scoring pipeline:
  1. Look up customer behavioral profile (B1)
  2. Compute behavioral deviation features (B1)
  3. Compute temporal / velocity features (B2)
  4. Score with LightGBM tabular model (B3)
  5. Score with IsolationForest anomaly model (B4)
  6. Generate SHAP-based reason codes (B7)
  7. Apply response policy (B8)
  8. Return calibrated risk score + decision + reason codes
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI

from api.schema import Decision, RiskResponse, TransactionContext
from features.behavioral import (
    build_customer_profiles,
    compute_behavioral_features,
)
from features.temporal import compute_temporal_features
from features.graph_features import build_transaction_graph, compute_graph_features
from models.fusion import simple_weighted_fusion
from models.explain import generate_reason_codes, generate_reason_codes_from_features
from policy.response import decide

logger = logging.getLogger("evo-pay")

app = FastAPI(
    title="EVO-PAY Blue Team",
    description="Adaptive Payment Fraud Defense System",
    version="0.5.0",
)

# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------
_profiles: dict = {}
_history_df: pd.DataFrame = pd.DataFrame()
_tabular_model: dict | None = None
_anomaly_model: dict | None = None
_transaction_graph = None
_data_loaded: bool = False

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "synthetic_transactions.csv"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


def _load_data() -> None:
    """Load data, build profiles, load trained models."""
    global _profiles, _history_df, _tabular_model, _anomaly_model, _transaction_graph, _data_loaded
    if _data_loaded:
        return

    # --- Transaction data + profiles ---
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        legit = df[df["attack_family"] == "legitimate"] if "attack_family" in df.columns else df
        _profiles = build_customer_profiles(legit)
        _history_df = legit.sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %d profiles, %d history rows", len(_profiles), len(_history_df))

        # Graph is built from the same legit history as temporal features (B5).
        # Note: because this history is legit-only, merchant_fraud_rate will
        # always read 0 here - there's no fraud label to propagate through the
        # graph from this data alone. shared_device_count, customer_degree and
        # dense-cluster signals are still meaningful from structure alone.
        _transaction_graph = build_transaction_graph(_history_df)
    except FileNotFoundError:
        logger.warning("No data at %s", DATA_PATH)

    # --- Trained tabular model ---
    try:
        from models.tabular import load_model
        _tabular_model = load_model("tabular_model")
        logger.info("Tabular model loaded: %s", _tabular_model["model_type"])
    except FileNotFoundError:
        logger.warning("No tabular model -- using heuristic scoring")

    # --- Trained anomaly model ---
    try:
        from models.anomaly import load_anomaly_model
        _anomaly_model = load_anomaly_model("anomaly_model")
        logger.info("Anomaly model loaded")
    except FileNotFoundError:
        logger.warning("No anomaly model")

    _data_loaded = True


@app.on_event("startup")
def startup():
    _load_data()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reload")
def reload_models():
    """Reload Blue models/data after an explicit adversarial retraining step."""
    global _data_loaded
    _data_loaded = False
    _load_data()
    return {
        "status": "reloaded",
        "profiles_loaded": len(_profiles),
        "history_rows": len(_history_df),
        "tabular_model": _tabular_model["model_type"] if _tabular_model else None,
        "anomaly_model": _anomaly_model is not None,
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "profiles_loaded": len(_profiles),
        "history_rows": len(_history_df),
        "tabular_model": _tabular_model["model_type"] if _tabular_model else None,
        "anomaly_model": _anomaly_model is not None,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_with_tabular(behavioral: dict, temporal: dict) -> float:
    """Score using trained tabular model."""
    from models.tabular import predict_tabular
    feature_names = _tabular_model["feature_names"]
    all_features = {**behavioral, **temporal}
    row = {fn: all_features.get(fn, 0.0) for fn in feature_names}
    X = pd.DataFrame([row])[feature_names]
    scores = predict_tabular(_tabular_model["model"], X, _tabular_model["model_type"])
    return float(np.clip(scores[0], 0.0, 1.0))


def _score_with_anomaly(behavioral: dict, temporal: dict) -> float:
    """Score using trained anomaly model."""
    from models.anomaly import predict_anomaly_single
    all_features = {**behavioral, **temporal}
    return predict_anomaly_single(_anomaly_model, all_features)


def _score_with_graph(ctx: TransactionContext, request_history: pd.DataFrame) -> tuple[float, dict]:
    """Graph structure score (B5). No trained graph model exists yet, so
    this is a hand-weighted combination of compute_graph_features() output -
    same "80% of value for 10% of effort" spirit as graph_features.py's own
    docstring, not a claim of a learned graph model.

    The current transaction's device/merchant edges - plus every edge from
    the caller's supplied customer_history - are added to a *copy* of the
    graph before scoring. Without this, an external caller (e.g. Red Team)
    evaluating one transaction at a time would only ever contribute a single
    edge per call, so shared_device_count would never reflect the rest of
    that customer's own campaign.

    Known limit: this graph is keyed by customer_id, so it can only ever
    catch structure within one customer's own accounts/devices. Fraud rings
    spanning multiple different customer_ids sharing a device would need a
    much larger shared graph across the whole ecosystem, which no caller
    currently supplies.
    """
    if _transaction_graph is None or _transaction_graph.number_of_nodes() == 0:
        return 0.0, {}

    graph = _transaction_graph.copy()
    cid = f"c:{ctx.customer_id}"

    if not request_history.empty:
        for _, row in request_history.iterrows():
            graph.add_edge(cid, f"d:{row.get('device_id', 'unknown')}")
            graph.add_edge(cid, f"m:{row.get('merchant_id', 'unknown')}")

    device_id = ctx.transaction.get("device_id", "unknown")
    merchant_id = ctx.transaction.get("merchant_id", "unknown")
    graph.add_edge(cid, f"d:{device_id}")
    graph.add_edge(cid, f"m:{merchant_id}")

    features = compute_graph_features(ctx.customer_id, graph, fraud_labels={})
    risk = (
        0.15 * features["shared_device_count"]
        + 0.50 * features["merchant_fraud_rate"]
        + 0.25 * features["is_part_of_dense_cluster"]
        + 0.02 * features["device_customer_degree"]
    )
    return round(min(1.0, risk), 4), features


def _temporal_signal_score(temporal: dict) -> float:
    """Diagnostic-only view of temporal/velocity behavior.

    The tabular model's feature_names already include the raw tx_count_*,
    amount_sum_*, velocity_ratio_* fields, so temporal signal is already
    inside tabular_score - it is NOT added again here as a separate fusion
    component (that would double-count the same evidence). This score
    exists only so model_scores.temporal is a real, if approximate,
    read on velocity for debugging/explainability, not a hardcoded 0.
    """
    if not temporal:
        return 0.0
    burst = min(1.0, temporal.get("amount_burst_ratio", 0.0) / 10.0)
    velocity = min(1.0, temporal.get("velocity_ratio_5min_1h", 0.0))
    return round(min(1.0, 0.6 * burst + 0.4 * velocity), 4)


def _score_heuristic(behavioral: dict) -> float:
    """Fallback heuristic score from behavioral features."""
    score = 0.0
    zscore = abs(behavioral.get("amount_zscore", 0))
    if zscore > 5: score += 0.35
    elif zscore > 3: score += 0.25
    elif zscore > 2: score += 0.15
    elif zscore > 1: score += 0.05
    if behavioral.get("is_new_device", 0): score += 0.20
    if behavioral.get("is_new_merchant_category", 0): score += 0.10
    dist = behavioral.get("location_distance_km", 0)
    if dist > 5000: score += 0.25
    elif dist > 1000: score += 0.15
    elif dist > 100: score += 0.08
    if behavioral.get("is_thin_file", 0): score += 0.10
    return min(score, 1.0)


def _generate_reason_codes(
    behavioral: dict,
    temporal: dict,
    tabular_score: float,
) -> list[str]:
    """Generate reason codes -- SHAP if available, else rule-based."""
    # Try SHAP-based codes if tabular model is available
    if _tabular_model is not None:
        try:
            feature_names = _tabular_model["feature_names"]
            all_features = {**behavioral, **temporal}
            row = {fn: all_features.get(fn, 0.0) for fn in feature_names}
            X = pd.DataFrame([row])[feature_names]
            codes = generate_reason_codes(
                _tabular_model["model"], X, _tabular_model["model_type"], top_n=3,
            )
            if codes and codes != ["model_prediction"]:
                return codes
        except Exception as e:
            logger.debug("SHAP failed, using rule-based: %s", e)

    # Fallback: rule-based from feature values
    return generate_reason_codes_from_features(behavioral, temporal)


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/evaluate", response_model=RiskResponse)
def evaluate(ctx: TransactionContext) -> RiskResponse:
    """Evaluate a transaction for fraud risk.

    Full pipeline: B1 + B2 features -> B3 tabular + B4 anomaly scoring
    -> B7 reason codes -> B8 response policy.
    """
    t0 = time.perf_counter()

    # --- 0. Resolve history: global + whatever this caller supplied ---
    # customer_history previously only fed the profile fallback (B1) and was
    # silently dropped for temporal features (B2), so an external caller
    # (e.g. the Red Team, whose customers aren't in Blue's own training data)
    # got real behavioral deviation scoring but empty/zero temporal features.
    request_history = pd.DataFrame(ctx.customer_history) if ctx.customer_history else pd.DataFrame()
    if not request_history.empty and "timestamp" in request_history.columns:
        request_history["timestamp"] = pd.to_datetime(request_history["timestamp"])
        combined_history = (pd.concat([_history_df, request_history], ignore_index=True)
                             if not _history_df.empty else request_history)
    else:
        combined_history = _history_df

    # --- 1. Customer profile (B1) ---
    profile = _profiles.get(ctx.customer_id)
    if profile is None and not request_history.empty:
        tmp_profiles = build_customer_profiles(request_history)
        profile = tmp_profiles.get(ctx.customer_id)

    # --- 2. Behavioral features (B1) ---
    behavioral = compute_behavioral_features(ctx.transaction, profile)

    # --- 3. Temporal features (B2) ---
    tx_ts = ctx.transaction.get("timestamp")
    if tx_ts and not combined_history.empty:
        temporal = compute_temporal_features(ctx.customer_id, tx_ts, combined_history)
    else:
        temporal = {}

    # --- 4. Tabular score (B3) ---
    if _tabular_model is not None:
        tabular_score = _score_with_tabular(behavioral, temporal)
    else:
        tabular_score = _score_heuristic(behavioral)

    # --- 5. Anomaly score (B4) ---
    if _anomaly_model is not None:
        anomaly_score = _score_with_anomaly(behavioral, temporal)
    else:
        anomaly_score = 0.0

    # --- 5b. Graph score (B5) ---
    graph_score, graph_features = _score_with_graph(ctx, request_history)

    # --- 5c. Temporal diagnostic score - not fused, see _temporal_signal_score ---
    temporal_score = _temporal_signal_score(temporal)

    # --- 6. Combined risk score (B6) ---
    # tabular already contains temporal features, so temporal is excluded here
    # to avoid double-counting the same evidence twice.
    risk_score = simple_weighted_fusion(
        {"tabular": tabular_score, "anomaly": anomaly_score, "graph": graph_score},
        weights={"tabular": 0.65, "anomaly": 0.25, "graph": 0.10},
    )
    # Safety net: never lower than either primary signal alone
    risk_score = max(risk_score, tabular_score * 0.85, anomaly_score * 0.5)
    risk_score = min(risk_score, 1.0)

    # --- 7. Decision (B8) ---
    decision = decide(risk_score)

    # --- 8. Reason codes (B7) ---
    reason_codes = _generate_reason_codes(behavioral, temporal, tabular_score)
    if not reason_codes:
        reason_codes = ["within_normal_behavior"]

    latency_ms = (time.perf_counter() - t0) * 1000

    return RiskResponse(
        risk_score=round(risk_score, 4),
        detected=risk_score >= 0.50,
        decision=decision,
        reason_codes=reason_codes,
        model_scores={
            "tabular": round(tabular_score, 4),
            "anomaly": round(anomaly_score, 4),
            "graph": graph_score,
            "temporal": temporal_score,
        },
    )
