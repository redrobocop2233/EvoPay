"""EVO-PAY Streamlit dashboard — Premium redesign.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

import pandas as pd
from eval.lineage import build_lineage_graph
import numpy as np
import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(
    page_title="EVO-PAY · Adaptive Fraud Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "synthetic_transactions.csv"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results" / "training_report.json"

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "red_and_blue_team"))

# ============================================================================
# Premium CSS Theme
# ============================================================================
st.markdown("""
<style>
/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Root variables ---------- */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.7);
    --bg-card-hover: rgba(30, 41, 59, 0.8);
    --border-subtle: rgba(99, 102, 241, 0.15);
    --border-glow: rgba(99, 102, 241, 0.35);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-indigo: #818cf8;
    --accent-violet: #a78bfa;
    --accent-cyan: #22d3ee;
    --accent-emerald: #34d399;
    --accent-rose: #fb7185;
    --accent-amber: #fbbf24;
    --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    --gradient-2: linear-gradient(135deg, #06b6d4, #3b82f6);
    --gradient-3: linear-gradient(135deg, #f43f5e, #ec4899);
    --radius: 16px;
    --radius-sm: 10px;
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.25), 0 0 0 1px var(--border-subtle);
    --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
    --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ---------- Global Dark Background ---------- */
.stApp, [data-testid="stAppViewContainer"],
.main .block-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}

html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    color: var(--text-primary) !important;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--text-primary) !important;
    background: rgba(99, 102, 241, 0.1) !important;
}

/* ---------- Headers ---------- */
h1 {
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: var(--gradient-1) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2.2rem !important;
    margin-bottom: 0.5rem !important;
}

h2 {
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    font-size: 1.5rem !important;
    margin-top: 2rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

h3 {
    font-weight: 600 !important;
    color: var(--accent-indigo) !important;
    font-size: 1.15rem !important;
}

/* ---------- Metric Cards ---------- */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
    padding: 20px 24px !important;
    box-shadow: var(--shadow-card) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stMetric"]:hover {
    border-color: var(--border-glow) !important;
    box-shadow: var(--shadow-glow) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
}

/* ---------- Buttons ---------- */
.stButton > button[kind="primary"],
.stButton > button {
    background: var(--gradient-1) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-family: var(--font-main) !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
}

/* ---------- DataFrames & Tables ---------- */
[data-testid="stDataFrame"],
.stDataFrame {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ---------- Charts ---------- */
[data-testid="stVegaLiteChart"] {
    background: var(--bg-card) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 12px !important;
}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

/* ---------- Select boxes, sliders, inputs ---------- */
[data-testid="stSelectbox"],
.stSelectbox > div > div {
    background: var(--bg-secondary) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

.stSlider [data-baseweb="slider"] {
    padding-top: 10px !important;
}

.stTextInput input, .stNumberInput input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-indigo) !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stSuccess {
    background: rgba(52, 211, 153, 0.1) !important;
    border-color: var(--accent-emerald) !important;
}

/* ---------- Spinner ---------- */
.stSpinner > div {
    border-top-color: var(--accent-indigo) !important;
}

/* ---------- JSON viewer ---------- */
[data-testid="stJson"] {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-subtle) !important;
    font-family: var(--font-mono) !important;
}

/* ---------- Info / Warning / Error ---------- */
div[data-testid="stAlert"] {
    backdrop-filter: blur(8px) !important;
    border-radius: var(--radius-sm) !important;
}

/* ---------- Progress bar ---------- */
.stProgress > div > div {
    background: var(--gradient-1) !important;
    border-radius: 8px !important;
}

/* ---------- Checkboxes ---------- */
.stCheckbox label {
    color: var(--text-secondary) !important;
}

/* ---------- Dividers ---------- */
hr {
    border-color: var(--border-subtle) !important;
    opacity: 0.5 !important;
}

/* ---------- Custom card class ---------- */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--shadow-card);
    margin-bottom: 16px;
}

.glass-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
}

/* ---------- Status badges ---------- */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.status-online { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
.status-offline { background: rgba(251, 113, 133, 0.15); color: #fb7185; border: 1px solid rgba(251, 113, 133, 0.3); }
.status-warning { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }

/* ---------- Animated gradient border on hero section ---------- */
.hero-section {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.05));
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient-1);
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    padding: 8px 20px !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: var(--accent-indigo) !important;
    color: var(--accent-indigo) !important;
}

/* ---------- Responsive columns (smaller gap) ---------- */
[data-testid="column"] {
    padding: 0 8px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 16px 0 8px;">
    <span style="font-size: 2rem;">🛡️</span><br/>
    <span style="font-size: 1.4rem; font-weight: 800; background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">EVO-PAY</span><br/>
    <span style="font-size: 0.75rem; color: #94a3b8; letter-spacing: 0.12em; text-transform: uppercase;">
        Adaptive Fraud Defense
    </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "⚡  Command Center",
    "🎯  Live Scoring",
    "📊  Model Performance",
    "🔬  Attack Families",
    "📋  Transaction Explorer",
    "🔄  Adaptive Closed Loop",
    "🧬  Red Team Evolution",
], label_visibility="collapsed")


def check_api():
    """Check if the API is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.json()
    except Exception:
        return None


def score_transaction(payload):
    """Send a transaction to the API for scoring."""
    try:
        r = requests.post(f"{API_URL}/evaluate", json=payload, timeout=15)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ======================================================================
# Page: Command Center (NEW — overview dashboard)
# ======================================================================
if page == "⚡  Command Center":
    st.title("Command Center")
    st.markdown("""
<div class="hero-section">
    <span style="font-size: 0.85rem; color: #94a3b8;">
        <strong style="color: #818cf8;">EVO-PAY</strong> is an adaptive adversarial defense system.
        Red Team attacks evolve through genetic algorithms. Blue Team detection retrains every generation.
        GenAI discovers novel threats and autopsies evasion failures.
    </span>
</div>
""", unsafe_allow_html=True)

    # System status
    health = check_api()
    c1, c2, c3, c4 = st.columns(4)
    if health:
        c1.metric("🟢 API Status", "Online")
        c2.metric("Profiles", health.get("profiles_loaded", 0))
        model_name = health.get("tabular_model") or health.get("model_type") or "Heuristic"
        c3.metric("Model", model_name)
        c4.metric("History", f"{health.get('history_rows', 0):,} rows")
    else:
        c1.metric("🔴 API Status", "Offline")
        c2.metric("Profiles", "—")
        c3.metric("Model", "—")
        c4.metric("History", "—")

    # Load latest closed loop results if available
    cl_path = Path(__file__).resolve().parent.parent / "integration" / "results" / "closed_loop_summary.json"
    if cl_path.exists():
        with open(cl_path) as f:
            cl = json.load(f)

        st.markdown("---")
        st.subheader("Latest Closed-Loop Results")

        orb = cl.get("overall_red_blue", {})
        bm = cl.get("blue_classification", {})
        lat = cl.get("latency", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Detection Rate", f"{orb.get('detection_rate', 0):.1%}")
        c2.metric("Attack Success", f"{orb.get('attack_success_rate', 0):.1%}")
        c3.metric("Blue F1", f"{bm.get('f1', 0):.3f}")
        c4.metric("Blue FPR", f"{bm.get('fpr', 0):.1%}")
        c5.metric("Latency (p50)", f"{lat.get('p50_ms', 0):.0f}ms")

        # Timeline chart
        timeline = pd.DataFrame(cl.get("generation_timeline", []))
        if not timeline.empty:
            timeline = timeline.set_index("generation")

            # Two-curve chart if available
            if "static_detection_rate" in timeline.columns and "adaptive_detection_rate" in timeline.columns:
                st.markdown("#### 🎯 Core Thesis: Static vs. Adaptive Defense")
                two_curve_df = timeline[["static_detection_rate", "adaptive_detection_rate"]].rename(
                    columns={
                        "static_detection_rate": "Static Defense",
                        "adaptive_detection_rate": "Adaptive Defense",
                    }
                )
                st.line_chart(two_curve_df)
                st.caption("Static line declining = Red attacks genuinely evolve past fixed defenses. "
                           "Adaptive line holding = Blue retraining keeps pace.")
            else:
                st.line_chart(timeline[["detection_rate", "attack_success_rate"]])

    else:
        st.info("No closed-loop results found. Navigate to **Adaptive Closed Loop** to run the adversarial evaluation.")


# ======================================================================
# Page: Live Scoring
# ======================================================================
elif page == "🎯  Live Scoring":
    st.title("Live Transaction Scoring")

    # API status
    health = check_api()
    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("API Status", "🟢 Online")
        col2.metric("Customer Profiles", health.get("profiles_loaded", 0))
        model_name = health.get("tabular_model") or health.get("model_type") or "Heuristic"
        col3.metric("Active Model", model_name)
    else:
        st.error("⚠️ Blue Team API is offline. Start with: `uvicorn api.main:app --port 8000`")

    st.markdown("---")

    # Real customers, not fabricated ones
    @st.cache_data
    def _load_sample_customers(n=25):
        if not DATA_PATH.exists():
            return pd.DataFrame()
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        legit = df[df["attack_family"] == "legitimate"] if "attack_family" in df.columns else df
        sample_ids = legit["customer_id"].drop_duplicates().head(n).tolist()
        return legit[legit["customer_id"].isin(sample_ids)]

    sample_df = _load_sample_customers()

    st.subheader("Submit a Transaction")

    if sample_df.empty:
        st.warning("No data found — run `python -m data.generate_synthetic` first.")
        customer_id = st.text_input("Customer ID", "")
        typical_amount, home_lat, home_lon, known_device = 100.0, 0.0, 0.0, ""
    else:
        customer_ids = sorted(sample_df["customer_id"].unique().tolist())
        customer_id = st.selectbox("Customer", customer_ids)
        customer_rows = sample_df[sample_df["customer_id"] == customer_id]
        typical_amount = float(customer_rows["amount"].mean())
        home_lat = float(customer_rows["location_lat"].mean())
        home_lon = float(customer_rows["location_lon"].mean())
        known_device = customer_rows["device_id"].mode().iloc[0]
        st.caption(f"Typical amount ~${typical_amount:.2f} · home ~({home_lat:.2f}, {home_lon:.2f}) "
                   f"· usual device `{known_device}`")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input("Amount ($)", min_value=0.01, value=round(typical_amount, 2), step=10.0)
        merchant_cat = st.selectbox("Merchant Category", [
            "grocery", "gas_station", "electronics", "restaurant", "online_retail",
            "apparel", "entertainment", "travel", "health", "jewelry",
        ])
        device_id = st.text_input("Device ID", known_device or "DEV_new")

    with col2:
        location_lat = st.number_input("Latitude", value=round(home_lat, 4), format="%.4f")
        location_lon = st.number_input("Longitude", value=round(home_lon, 4), format="%.4f")
        timestamp = st.text_input("Timestamp", "2026-08-01T14:00:00")
        campaign_id = st.text_input("Campaign ID", "demo")

    if st.button("⚡ Evaluate Transaction", type="primary"):
        payload = {
            "campaign_id": campaign_id,
            "customer_id": customer_id,
            "transaction": {
                "amount": amount,
                "merchant_category": merchant_cat,
                "device_id": device_id,
                "location_lat": location_lat,
                "location_lon": location_lon,
                "timestamp": timestamp,
            },
        }

        with st.spinner("Scoring..."):
            result = score_transaction(payload)

        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            st.markdown("---")
            st.subheader("Scoring Result")

            decision = result.get("decision", "unknown")
            risk_score = result.get("risk_score", 0)
            detected = result.get("detected", False)

            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score:.4f}")
            col2.metric("Decision", decision.upper())
            col3.metric("Flagged", "🚨 YES" if detected else "✅ NO")

            # Risk gauge
            st.progress(min(risk_score, 1.0))

            # Reason codes
            reason_codes = result.get("reason_codes", [])
            if reason_codes:
                st.subheader("Reason Codes")
                for code in reason_codes:
                    st.markdown(f"- `{code}`")

            # Model scores breakdown
            scores = result.get("model_scores", {})
            if scores:
                st.subheader("Model Scores Breakdown")
                score_df = pd.DataFrame([scores])
                st.bar_chart(score_df.T.rename(columns={0: "Score"}))

            with st.expander("📦 Raw API Response"):
                st.json(result)

    # Quick demo scenarios
    st.markdown("---")
    st.subheader("Quick Scenarios")
    st.caption(f"Built around **{customer_id}** — deviations from their real baseline.")

    scenarios = {
        "✅ Normal": {
            "campaign_id": "demo", "customer_id": customer_id,
            "transaction": {"amount": round(typical_amount, 2), "merchant_category": "gas_station",
                           "device_id": known_device or "DEV_new", "location_lat": round(home_lat, 4),
                           "location_lon": round(home_lon, 4), "timestamp": "2026-07-30T15:57:57"},
        },
        "🚨 Account Takeover": {
            "campaign_id": "demo", "customer_id": customer_id,
            "transaction": {"amount": round(typical_amount * 15, 2), "merchant_category": "electronics",
                           "device_id": "DEV_STOLEN", "location_lat": round(home_lat, 2) + 10,
                           "location_lon": round(home_lon, 2) + 10, "timestamp": "2026-08-01T03:15:00"},
        },
        "👤 Synthetic Identity": {
            "campaign_id": "demo", "customer_id": "BRAND_NEW_" + customer_id[-6:],
            "transaction": {"amount": 4200.00, "merchant_category": "jewelry",
                           "device_id": "DEV_UNKNOWN", "location_lat": round(home_lat, 2),
                           "location_lon": round(home_lon, 2), "timestamp": "2026-08-01T02:00:00"},
        },
    }

    cols = st.columns(len(scenarios))
    for (name, payload), col in zip(scenarios.items(), cols):
        with col:
            if st.button(name, use_container_width=True):
                result = score_transaction(payload)
                if "error" not in result:
                    st.metric("Score", f"{result['risk_score']:.3f}")
                    st.metric("Decision", result['decision'].upper())
                    for code in result.get("reason_codes", []):
                        st.caption(f"- {code}")


# ======================================================================
# Page: Model Performance
# ======================================================================
elif page == "📊  Model Performance":
    st.title("Model Performance Report")

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            report = json.load(f)

        st.markdown(f"""
<div class="hero-section">
    <span style="font-size: 1.1rem; font-weight: 700; color: #818cf8;">
        {report['best_model']}
    </span><br/>
    <span style="color: #94a3b8; font-size: 0.85rem;">
        Best model · {report['n_features']} features
    </span>
</div>
""", unsafe_allow_html=True)

        # Test metrics
        st.subheader("Test Set Metrics")
        tm = report["test_metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PR-AUC", f"{tm.get('pr_auc', 0):.4f}")
        col2.metric("Precision", f"{tm.get('precision', 0):.4f}")
        col3.metric("Recall", f"{tm.get('recall', 0):.4f}")
        col4.metric("F1", f"{tm.get('f1', 0):.4f}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.4f}")
        col2.metric("FPR", f"{tm.get('fpr', 0):.4f}")
        col3.metric("Brier Score", f"{tm.get('brier_score', 0):.4f}")
        col4.metric("Recall@1%FPR", f"{tm.get('recall_at_1pct_fpr', 0):.4f}")

        # Model comparison table
        val_metrics = report.get("validation_metrics", {})
        if val_metrics:
            st.subheader("Model Comparison (Validation)")
            rows = []
            for model_name, metrics in val_metrics.items():
                rows.append({
                    "Model": model_name,
                    "PR-AUC": metrics.get("pr_auc", 0),
                    "ROC-AUC": metrics.get("roc_auc", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1": metrics.get("f1", 0),
                    "FPR": metrics.get("fpr", 0),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    else:
        st.warning("No training report found. Run: `python -m models.train_pipeline`")


# ======================================================================
# Page: Attack Family Analysis
# ======================================================================
elif page == "🔬  Attack Families":
    st.title("Attack Family Analysis")

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            report = json.load(f)

        breakdown = report.get("attack_family_breakdown", {})
        if breakdown:
            rows = []
            for family, stats in breakdown.items():
                rows.append({
                    "Attack Family": family,
                    "Count": stats.get("count", 0),
                    "Recall": stats.get("recall", 0),
                    "Mean Score": stats.get("mean_score", 0),
                })
            df_atk = pd.DataFrame(rows).set_index("Attack Family")
            st.dataframe(df_atk, use_container_width=True)
            st.bar_chart(df_atk["Recall"])
        else:
            st.info("No attack family breakdown available.")
    else:
        st.warning("No training report found.")


# ======================================================================
# Page: Transaction Explorer
# ======================================================================
elif page == "📋  Transaction Explorer":
    st.title("Transaction Explorer")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")
        col3.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
        col4.metric("Date Range", f"{df['timestamp'].dt.date.nunique()} days")

        st.subheader("Attack Family Distribution")
        family_counts = df["attack_family"].value_counts()
        st.bar_chart(family_counts)

        st.subheader("Browse Transactions")
        family_filter = st.multiselect(
            "Filter by Attack Family",
            options=df["attack_family"].unique().tolist(),
            default=["legitimate"],
        )
        filtered = df[df["attack_family"].isin(family_filter)]
        st.dataframe(filtered.head(100), use_container_width=True)
    else:
        st.warning("No data found. Run: `python -m data.generate_synthetic`")


# ======================================================================
# Page: Adaptive Closed Loop
# ======================================================================
elif page == "🔄  Adaptive Closed Loop":
    st.title("Adaptive Closed Loop")
    st.markdown("""
<div class="hero-section">
    <span style="color: #94a3b8; font-size: 0.85rem;">
        <strong style="color: #818cf8;">GenAI discovers</strong> →
        <strong style="color: #fb7185;">Red evolves</strong> →
        <strong style="color: #22d3ee;">Blue detects</strong> →
        <strong style="color: #818cf8;">GenAI autopsies</strong> →
        <strong style="color: #fb7185;">Red mutates</strong> →
        <strong style="color: #22d3ee;">Blue retrains</strong>
    </span>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    generations = c1.slider("Generations", 1, 10, 3)
    population = c2.slider("Population", 6, 40, 12, step=2)
    discover = c3.slider("Initial GenAI ideas", 1, 8, 4)
    retrain_every = c4.selectbox("Blue retrain", [0, 1, 2, 3], format_func=lambda x: "Disabled" if x == 0 else f"Every {x} gen(s)")
    use_live_blue = st.checkbox("Use Blue Team API", value=True)
    use_genai = st.checkbox("Use Gemini", value=True)

    if st.button("🚀 Run Adaptive Loop", type="primary"):
        from integration.closed_loop import run_loop
        with st.spinner("Running adversarial loop — this may take a while..."):
            try:
                controller, discoveries, autopsies, summary = run_loop(
                    generations=generations,
                    population=population,
                    discover=discover,
                    api_url=API_URL if use_live_blue else "",
                    use_genai=use_genai,
                    retrain_blue_every=retrain_every,
                )
                st.session_state["closed_loop_summary"] = summary
                st.session_state["closed_loop_discoveries"] = discoveries
                st.session_state["closed_loop_autopsies"] = autopsies
                st.success("✅ Closed loop completed successfully.")
            except Exception as e:
                st.exception(e)

    summary = st.session_state.get("closed_loop_summary")
    if summary:
        bm = summary.get("blue_classification", {})
        orb = summary.get("overall_red_blue", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Blue Recall", f"{bm.get('recall', 0):.1%}")
        c2.metric("Blue FPR", f"{bm.get('fpr', 0):.1%}")
        c3.metric("Blue F1", f"{bm.get('f1', 0):.3f}")
        c4.metric("Attack Success", f"{orb.get('attack_success_rate', 0):.1%}")
        c5.metric("GenAI Discoveries", summary.get("discoveries", 0))

        tta = summary.get("time_to_adapt", {})
        if tta.get("avg_time_to_adapt_generations") is not None:
            st.metric("⚡ Avg Time-to-Adapt", f"{tta['avg_time_to_adapt_generations']} generations")

        timeline = pd.DataFrame(summary.get("generation_timeline", []))
        if not timeline.empty:
            timeline = timeline.set_index("generation")

            # Tab layout for charts
            tab_thesis, tab_evolution, tab_lineage, tab_data = st.tabs(["🎯 Core Thesis", "📈 Evolution", "🧬 Lineage", "📋 Data"])

            with tab_thesis:
                if "static_detection_rate" in timeline.columns and "adaptive_detection_rate" in timeline.columns:
                    st.markdown("""
**Static vs. Adaptive Defense** — the key proof of concept.
- 📉 **Static defense declining** = Red attacks genuinely evolve past a fixed model.
- 📈 **Adaptive defense holding** = Blue retraining loop is working.
""")
                    two_curve_df = timeline[["static_detection_rate", "adaptive_detection_rate"]].rename(
                        columns={
                            "static_detection_rate": "Static Defense",
                            "adaptive_detection_rate": "Adaptive Defense",
                        }
                    )
                    st.line_chart(two_curve_df)
                else:
                    st.info("Run with `--local-blue` (TrainableDetector) to see the two-curve thesis chart.")

            with tab_evolution:
                st.line_chart(timeline[["detection_rate", "attack_success_rate", "mean_fitness"]])

            with tab_lineage:
                mem_path = results_dir / "strategy_memory.csv"
                if mem_path.exists():
                    mem = pd.read_csv(mem_path)
                    graph = build_lineage_graph(mem)
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plot_ok = True
                    try:
                        from eval.lineage import plot_lineage_tree
                        plot_lineage_tree(graph, ax=ax)
                        st.pyplot(fig)
                    except Exception as exc:
                        plot_ok = False
                        st.warning(f"Lineage plot unavailable: {exc}")
                    cols = [c for c in ["campaign_id","parent_campaign_id","generation","attack_family","fitness","detection_probability","mutation_summary"] if c in mem.columns]
                    st.dataframe(mem[cols], use_container_width=True)
                else:
                    st.info("Run the closed loop to populate lineage data.")

            with tab_data:
                st.dataframe(timeline, use_container_width=True)

        # Latency
        latency = summary.get("latency", {})
        if latency:
            st.subheader("⏱ Inference Latency")
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("p50", f"{latency.get('p50_ms', 0):.1f}ms")
            lc2.metric("p95", f"{latency.get('p95_ms', 0):.1f}ms")
            lc3.metric("Mean", f"{latency.get('mean_ms', 0):.1f}ms")

        # Blue classification detail
        with st.expander("📊 Blue Classification Detail"):
            st.json(bm)

        # GenAI Autopsy Trace
        st.subheader("🧬 GenAI Autopsy → Mutation Trace")
        autopsies = st.session_state.get("closed_loop_autopsies", [])
        if autopsies:
            autopsy_rows = []
            for i, a in enumerate(autopsies):
                mutations = a.get("recommended_mutations", [])
                autopsy_rows.append({
                    "strategy": a.get("strategy_id", ""),
                    "detected": "🚨" if a.get("detected") else "✅ Evaded",
                    "risk": round(a.get("blue_risk_score", 0), 3),
                    "weakest_signal": a.get("weakest_signal", ""),
                    "mutations": ", ".join(
                        m.get("dimension", "") + " " + ("↑" if m.get("direction") in ("increase", "add") else "↓")
                        for m in mutations
                    ),
                    "confidence": a.get("confidence", ""),
                })
            st.dataframe(pd.DataFrame(autopsy_rows), use_container_width=True)

            # Detailed trace
            selected_idx = st.selectbox(
                "Select autopsy to trace", range(len(autopsies)),
                format_func=lambda i: f"#{i+1}: {autopsies[i].get('strategy_id', 'unknown')}"
            )
            if selected_idx is not None:
                a = autopsies[selected_idx]
                tc1, tc2, tc3 = st.columns(3)

                with tc1:
                    st.markdown("#### 🔴 Attack")
                    st.write(f"Strategy: `{a.get('strategy_id', '')}`")
                    st.write(f"Detected: {'🚨 Yes' if a.get('detected') else '✅ No'}")
                    st.write(f"Risk score: `{a.get('blue_risk_score', 0):.3f}`")

                with tc2:
                    st.markdown("#### 🔵 Defense")
                    reason_codes = a.get("reason_codes", [])
                    if reason_codes:
                        for code in reason_codes:
                            st.write(f"- `{code}`")
                    else:
                        st.write("No specific reason codes")
                    st.write(f"Weakest: `{a.get('weakest_signal', 'unknown')}`")

                with tc3:
                    st.markdown("#### 🧬 Autopsy")
                    st.write(a.get("explanation", "No explanation available"))

                mutations = a.get("recommended_mutations", [])
                if mutations:
                    st.markdown("**Genome Mutation Diff**")
                    diff_data = []
                    for m in mutations:
                        direction = m.get("direction", "")
                        diff_data.append({
                            "Dimension": m.get("dimension", ""),
                            "Direction": "↑ increase" if direction in ("increase", "add") else "↓ decrease",
                            "Change": "+0.10" if direction in ("increase", "add") else "−0.10",
                            "Rationale": m.get("rationale", ""),
                        })
                    st.dataframe(pd.DataFrame(diff_data), use_container_width=True)
        else:
            st.info("No autopsies — run with **Gemini enabled** to see GenAI feedback.")

        with st.expander("📖 Metric Definitions"):
            st.markdown("""
| Metric | Definition |
|--------|-----------|
| **Detection rate** | Fraction of Red campaigns caught by Blue |
| **Attack success** | Fraction of Red campaigns that evade Blue |
| **FPR** | Fraction of untouched legitimate transactions flagged |
| **Static defense** | Detection by the frozen generation-0 Blue model |
| **Adaptive defense** | Detection by the continuously retrained Blue model |
| **Genome diversity** | Pairwise distance across Red behavioral vectors |
| **Fidelity proxy** | Simulator-quality diagnostic (not real-world prevalence) |
""")


# ======================================================================
# Page: Red Team Evolution
# ======================================================================
elif page == "🧬  Red Team Evolution":
    st.title("Red Team Evolution")
    st.markdown("Run the evolutionary red team live against the Blue Team and observe adaptation in real-time.")

    api_status = check_api()
    if api_status is None:
        st.error("Blue Team API not reachable. Start with: `python -m uvicorn api.main:app --port 8000`")
    else:
        st.success(f"🟢 Connected — {api_status.get('profiles_loaded', 0)} profiles, "
                   f"tabular model {'loaded' if api_status.get('tabular_model') else 'NOT loaded (heuristic fallback)'}")

    col1, col2, col3 = st.columns(3)
    num_customers = col1.slider("Ecosystem customers", 20, 300, 100, step=20)
    population_size = col2.slider("Population size", 5, 50, 20, step=5)
    generations = col3.slider("Generations", 1, 20, 8)

    use_fallback = st.checkbox("Fall back to local heuristic if API unreachable", value=True)

    if st.button("🧬 Run Evolution", type="primary", disabled=(api_status is None and not use_fallback)):
        with st.spinner(f"Running {generations} generations × {population_size} strategies..."):
            from red_and_blue_team.ecosystem import PaymentEcosystem
            from red_and_blue_team.blue_team_client import BlueTeamClient
            from red_and_blue_team.blue_team import HeuristicDetector
            from red_and_blue_team.red_team import RedTeamController, FailureAnalyzer

            eco = PaymentEcosystem(num_customers=num_customers, num_merchants=max(15, num_customers // 3), num_days=60)
            eco.generate_transactions()

            detector = BlueTeamClient(base_url=API_URL, fallback=HeuristicDetector() if use_fallback else None)
            controller = RedTeamController(eco, detector=detector, population_size=population_size)
            memory = controller.evolve(generations=generations)

            st.session_state["evo_stats"] = controller.generation_stats
            st.session_state["evo_memory"] = memory
            st.session_state["evo_analyzer"] = FailureAnalyzer().analyze(memory)

    if "evo_stats" in st.session_state:
        stats_df = pd.DataFrame(st.session_state["evo_stats"]).set_index("generation")

        st.subheader("Generation Trends")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(stats_df[["detection_rate"]])
            st.caption("Fraction of campaigns Blue caught per generation")
        with c2:
            st.line_chart(stats_df[["avg_fitness"]])
            st.caption("Average Red fitness per generation")

        st.subheader("Blind Spot Analysis")
        dim_df = pd.DataFrame(st.session_state["evo_analyzer"]).T
        dim_df = dim_df[dim_df["active"] > 0].sort_values("detection_rate")
        st.dataframe(dim_df, use_container_width=True)
        if len(dim_df) > 0:
            lowest = dim_df.index[0]
            st.caption(f"Lowest detection: **{lowest}** "
                       f"({dim_df.loc[lowest, 'detection_rate']:.0%}) — primary blind spot")

        st.subheader("Top Strategies by Fitness")
        memory = st.session_state["evo_memory"]
        top = memory.top(10)
        top_df = pd.DataFrame([{
            "gen": r.generation, "fitness": round(r.fitness, 4), "gain": round(r.gain, 2),
            "risk": round(r.risk_score, 3), "detected": "🚨" if r.detected else "✅",
            "dimensions": ", ".join(r.active_dimensions) or "none",
        } for r in top])
        st.dataframe(top_df, use_container_width=True)

        with st.expander("📖 Reading this page"):
            st.markdown("""
- **Detection rate** near 100% doesn't always mean Red can't evade — it can also mean
  Red's synthetic economy sits outside the distribution Blue was trained on.
- **Blind-spot table**: a dimension with many attempts and low detection is where
  Red finds genuine leverage.
- Each campaign triggers multiple API calls, so larger settings take longer.
""")
