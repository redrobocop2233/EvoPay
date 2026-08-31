"""Streamlit dashboard for EVO-PAY Blue Team demo.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time

import pandas as pd
import numpy as np
import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(
    page_title="EVO-PAY",
    page_icon="🛡️",
    layout="wide",
)

API_URL = "http://localhost:8000"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "synthetic_transactions.csv"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results" / "training_report.json"


import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "red_team"))

# --- Sidebar ---
st.sidebar.title("EVO-PAY")
st.sidebar.markdown("**Adaptive Fraud Defense**")
page = st.sidebar.radio("Navigate", [
    "Live Scoring",
    "Model Performance",
    "Attack Family Analysis",
    "Transaction Explorer",
    "Adaptive Closed Loop",
    "Red Team Evolution",
])


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
# Page: Live Scoring
# ======================================================================
if page == "Live Scoring":
    st.title("Live Transaction Scoring")

    # API status
    health = check_api()
    if health:
        col1, col2, col3 = st.columns(3)
        col1.metric("API Status", "Online")
        col2.metric("Customer Profiles", health.get("profiles_loaded", 0))
        model_name = health.get("tabular_model") or health.get("model_type") or "Heuristic"
        col3.metric("Model", model_name)
    else:
        st.error("API is offline. Start with: `uvicorn api.main:app --port 8000`")

    st.markdown("---")

    # Real customers, not fabricated ones - a hardcoded fake customer_id
    # doesn't match any loaded profile, and a hardcoded lat/lon doesn't
    # correspond to that (nonexistent) customer's actual home location.
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
        st.warning("No data found - run data.generate_synthetic first. Falling back to manual entry only.")
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
        st.caption(f"Typical amount ~${typical_amount:.2f} | home ~({home_lat:.2f}, {home_lon:.2f}) "
                   f"| usual device {known_device}")

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

    if st.button("Evaluate Transaction", type="primary"):
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
            st.subheader("Result")

            # Decision display with color
            decision = result.get("decision", "unknown")
            risk_score = result.get("risk_score", 0)
            detected = result.get("detected", False)

            decision_colors = {
                "allow": "green", "challenge": "orange",
                "hold": "red", "block": "red",
            }
            color = decision_colors.get(decision, "gray")

            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score:.4f}")
            col2.metric("Decision", decision.upper())
            col3.metric("Detected", "YES" if detected else "NO")

            # Risk gauge
            st.progress(min(risk_score, 1.0))

            # Reason codes
            st.subheader("Reason Codes")
            for code in result.get("reason_codes", []):
                st.markdown(f"- **{code}**")

            # Model scores breakdown
            st.subheader("Model Scores")
            scores = result.get("model_scores", {})
            score_df = pd.DataFrame([scores])
            st.bar_chart(score_df.T.rename(columns={0: "Score"}))

            # Raw JSON
            with st.expander("Raw Response"):
                st.json(result)

    # Quick demo scenarios
    st.markdown("---")
    st.subheader("Quick Demo Scenarios")
    st.caption(f"Built around the selected customer ({customer_id}) so 'normal' actually reflects "
               f"their real history, and the attack scenarios deviate from *their* real baseline.")

    scenarios = {
        "Normal Transaction": {
            "campaign_id": "demo", "customer_id": customer_id,
            "transaction": {"amount": round(typical_amount, 2), "merchant_category": "gas_station",
                           "device_id": known_device or "DEV_new", "location_lat": round(home_lat, 4),
                           "location_lon": round(home_lon, 4), "timestamp": "2026-07-30T15:57:57"},
        },
        "Account Takeover": {
            "campaign_id": "demo", "customer_id": customer_id,
            "transaction": {"amount": round(typical_amount * 15, 2), "merchant_category": "electronics",
                           "device_id": "DEV_STOLEN", "location_lat": round(home_lat, 2) + 10,
                           "location_lon": round(home_lon, 2) + 10, "timestamp": "2026-08-01T03:15:00"},
        },
        "Synthetic Identity": {
            "campaign_id": "demo", "customer_id": "BRAND_NEW_" + customer_id[-6:],
            "transaction": {"amount": 4200.00, "merchant_category": "jewelry",
                           "device_id": "DEV_UNKNOWN", "location_lat": round(home_lat, 2),
                           "location_lon": round(home_lon, 2), "timestamp": "2026-08-01T02:00:00"},
        },
    }

    cols = st.columns(len(scenarios))
    for (name, payload), col in zip(scenarios.items(), cols):
        with col:
            if st.button(name):
                result = score_transaction(payload)
                if "error" not in result:
                    st.metric("Score", f"{result['risk_score']:.3f}")
                    st.metric("Decision", result['decision'].upper())
                    for code in result.get("reason_codes", []):
                        st.caption(f"- {code}")


# ======================================================================
# Page: Model Performance
# ======================================================================
elif page == "Model Performance":
    st.title("Model Performance Report")

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            report = json.load(f)

        st.subheader("Best Model")
        st.info(f"**{report['best_model']}** with {report['n_features']} features")

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
        st.subheader("Model Comparison (Validation)")
        val_metrics = report.get("validation_metrics", {})
        if val_metrics:
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
elif page == "Attack Family Analysis":
    st.title("Per-Attack-Family Detection")

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

            # Bar chart of recall by family
            st.bar_chart(df_atk["Recall"])
        else:
            st.info("No attack family breakdown available.")
    else:
        st.warning("No training report found.")


# ======================================================================
# Page: Transaction Explorer
# ======================================================================
elif page == "Transaction Explorer":
    st.title("Transaction Data Explorer")

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")
        col3.metric("Unique Customers", f"{df['customer_id'].nunique():,}")

        # Attack family breakdown
        st.subheader("Attack Family Distribution")
        family_counts = df["attack_family"].value_counts()
        st.bar_chart(family_counts)

        # Filter and browse
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

elif page == "Adaptive Closed Loop":
    st.title("Adaptive Red / Blue Closed Loop")
    st.markdown("**Gemini discovers → Red evolves → Blue detects → Gemini autopsies → Red mutates → Blue can retrain.**")

    c1, c2, c3, c4 = st.columns(4)
    generations = c1.slider("Generations", 1, 10, 3)
    population = c2.slider("Population", 6, 40, 12, step=2)
    discover = c3.slider("Initial GenAI ideas", 1, 8, 4)
    retrain_every = c4.selectbox("Blue retrain", [0, 1, 2, 3], format_func=lambda x: "Disabled" if x == 0 else f"Every {x} generation(s)")
    use_live_blue = st.checkbox("Use Blue Team API", value=True)
    use_genai = st.checkbox("Use Gemini", value=True)

    if st.button("Run Adaptive Loop", type="primary"):
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
                st.success("Closed loop completed.")
            except Exception as e:
                st.exception(e)

    summary = st.session_state.get("closed_loop_summary")
    if summary:
        bm = summary.get("blue_classification", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Blue Recall", f"{bm.get('recall', 0):.1%}")
        c2.metric("Blue FPR", f"{bm.get('fpr', 0):.1%}")
        c3.metric("Red Attack Success", f"{summary['overall_red_blue'].get('attack_success_rate', 0):.1%}")
        c4.metric("GenAI Discoveries", summary.get("discoveries", 0))

        timeline = pd.DataFrame(summary.get("generation_timeline", []))
        if not timeline.empty:
            timeline = timeline.set_index("generation")
            st.subheader("Adversarial co-evolution")
            st.line_chart(timeline[["detection_rate", "attack_success_rate", "mean_fitness"]])
            st.dataframe(timeline, use_container_width=True)

        st.subheader("Blue performance")
        st.json(bm)

        st.subheader("GenAI feedback")
        autopsies = st.session_state.get("closed_loop_autopsies", [])
        if autopsies:
            st.dataframe(pd.DataFrame([{
                "strategy": a.get("strategy_id"),
                "detected": a.get("detected"),
                "risk": a.get("blue_risk_score"),
                "weakest_signal": a.get("weakest_signal"),
                "mutations": ", ".join(m.get("dimension", "") + ":" + m.get("direction", "") for m in a.get("recommended_mutations", [])),
                "confidence": a.get("confidence"),
            } for a in autopsies]), use_container_width=True)

        with st.expander("Metric definitions"):
            st.markdown("""
- **Detection rate**: fraction of Red campaigns caught by Blue.
- **Attack success**: fraction of Red campaigns that evade Blue.
- **FPR**: fraction of untouched legitimate transactions flagged by Blue.
- **Genome diversity**: diversity of Red behavioral combinations.
- **Fidelity proxy**: simulator-quality diagnostic; it is not a claim about real-world fraud prevalence.
- **Holdout attacks**: fresh genomes evaluated without GenAI feedback.
""")


elif page == "Red Team Evolution":
    st.title("Red Team Evolution")
    st.markdown("Runs the evolutionary red team live against this Blue Team API and shows what it finds.")

    api_status = check_api()
    if api_status is None:
        st.error("Blue Team API is not reachable at " + API_URL + ". Start it with: "
                 "`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`")
    else:
        st.success(f"Connected - {api_status.get('profiles_loaded', 0)} profiles, "
                   f"{api_status.get('history_rows', 0)} history rows, "
                   f"tabular model {'loaded' if api_status.get('tabular_model') else 'NOT loaded (heuristic fallback)'}")

    col1, col2, col3 = st.columns(3)
    num_customers = col1.slider("Ecosystem customers", 20, 300, 100, step=20)
    population_size = col2.slider("Population size", 5, 50, 20, step=5)
    generations = col3.slider("Generations", 1, 20, 8)

    use_fallback = st.checkbox("Fall back to local heuristic detector if the API is unreachable", value=True)

    if st.button("Run evolution", type="primary", disabled=(api_status is None and not use_fallback)):
        with st.spinner(f"Running {generations} generations against {population_size} strategies each..."):
            from red_team.ecosystem import PaymentEcosystem
            from red_team.blue_team_client import BlueTeamClient
            from red_team.blue_team import HeuristicDetector
            from red_team.red_team import RedTeamController, FailureAnalyzer

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

        st.subheader("Generation trend")
        c1, c2 = st.columns(2)
        with c1:
            st.line_chart(stats_df[["detection_rate"]])
            st.caption("Fraction of campaigns Blue caught, per generation")
        with c2:
            st.line_chart(stats_df[["avg_fitness"]])
            st.caption("Average fitness of the population, per generation")

        st.subheader("Detection rate by active dimension (blind-spot signal)")
        dim_df = pd.DataFrame(st.session_state["evo_analyzer"]).T
        dim_df = dim_df[dim_df["active"] > 0].sort_values("detection_rate")
        st.dataframe(dim_df, use_container_width=True)
        if len(dim_df) > 0:
            lowest = dim_df.index[0]
            st.caption(f"Lowest detection rate this run: **{lowest}** "
                       f"({dim_df.loc[lowest, 'detection_rate']:.0%}) - the closest thing to a blind spot found.")

        st.subheader("Top strategies by fitness")
        memory = st.session_state["evo_memory"]
        top = memory.top(10)
        top_df = pd.DataFrame([{
            "generation": r.generation, "fitness": r.fitness, "gain": r.gain,
            "risk_score": r.risk_score, "detected": r.detected,
            "dimensions": ", ".join(r.active_dimensions) or "none",
        } for r in top])
        st.dataframe(top_df, use_container_width=True)

        with st.expander("Reading this page"):
            st.markdown("""
- **Detection rate** near 100% doesn't necessarily mean Red can't evade - it
  can also mean Red's synthetic economy sits outside the distribution Blue
  was trained/calibrated on, inflating risk for *legitimate* Red traffic too.
  Worth checking a known-legit Red transaction's risk_score independently
  before concluding the model is simply very good.
- **Blind-spot table**: a dimension with many attempts (`active`) and a low
  `detection_rate` is where the evolutionary search is finding real leverage.
- This page calls the live API once per transaction - a full run of
  `population_size x generations` campaigns means several times that many
  HTTP calls, so larger settings take longer.
""")
