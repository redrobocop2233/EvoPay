
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="EVO-PAY | Adaptive Fraud Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,.14), rgba(14,165,233,.08));
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem;}
    .hero p {margin: .45rem 0 0; opacity: .78;}
    .pill {
        display:inline-block; padding:.22rem .55rem; border-radius:999px;
        background:rgba(34,197,94,.13); color:#16a34a; font-size:.78rem;
        font-weight:600; margin-right:.3rem;
    }
    .stage {
        border:1px solid rgba(128,128,128,.25);
        border-radius:14px; padding:1rem; height:100%;
        background:rgba(128,128,128,.035);
    }
    .stage h3 {margin-top:0;}
    .muted {opacity:.68;}
    .small {font-size:.84rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.2);
        border-radius: 12px;
        padding: .55rem .7rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def run_closed_loop(generations: int, population: int, discover: int):
    cmd = [
        sys.executable, "-m", "integration.closed_loop",
        "--generations", str(generations),
        "--population", str(population),
        "--discover", str(discover),
    ]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=Path.cwd(),
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", "Closed-loop run timed out after 15 minutes."

def parse_results(text: str):
    generations = []
    pattern = re.compile(
        r"^\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*"
        r"([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)",
        re.M,
    )
    for m in pattern.finditer(text):
        generations.append({
            "generation": int(m.group(1)),
            "detection": float(m.group(2)),
            "attack_success": float(m.group(3)),
            "avg_fitness": float(m.group(4)),
            "avg_risk": float(m.group(5)),
            "families": int(m.group(6)),
            "diversity": float(m.group(7)),
        })

    metrics = {}
    metric_map = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "fpr": "fpr",
        "roc_auc": "roc_auc",
        "pr_auc": "pr_auc",
        "recall_at_1pct_fpr": "recall_at_1pct_fpr",
        "recall_at_5pct_fpr": "recall_at_5pct_fpr",
    }
    for line in text.splitlines():
        m = re.match(r"\s*([a-zA-Z0-9_]+)\s*:\s*([0-9.]+)", line)
        if m and m.group(1) in metric_map:
            metrics[m.group(1)] = float(m.group(2))

    discoveries = re.search(r"GenAI discoveries:\s*(\d+)", text)
    autopsies = re.search(r"GenAI autopsies:\s*(\d+)", text)

    return (
        pd.DataFrame(generations),
        metrics,
        int(discoveries.group(1)) if discoveries else None,
        int(autopsies.group(1)) if autopsies else None,
    )

def health_check(url: str):
    try:
        r = requests.get(url.rstrip("/") + "/health", timeout=3)
        return r.ok, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    except Exception as e:
        return False, str(e)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🛡️ EVO-PAY</h1>
  <p>Adaptive Red-Team / Blue-Team payment fraud laboratory</p>
  <span class="pill">IDENTIFY</span>
  <span class="pill">GENERATE</span>
  <span class="pill">ATTACK</span>
  <span class="pill">DEFEND</span>
  <span class="pill">ADAPT</span>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Experiment")
    generations = st.slider("Generations", 1, 20, 3)
    population = st.slider("Population", 4, 50, 12)
    discover = st.slider("GenAI discoveries", 1, 10, 4)

    st.divider()
    st.subheader("Blue Team API")
    health_url = st.text_input("Health endpoint", "http://127.0.0.1:8000")

    ok, health = health_check(health_url)
    if ok:
        st.success("Blue Team online")
    else:
        st.warning("Blue Team unavailable")

    st.divider()
    run = st.button("🚀 Run Adaptive Closed Loop", type="primary", use_container_width=True)

# ---------- Architecture ----------
st.subheader("Closed-loop architecture")
cols = st.columns(5)
stages = [
    ("01", "Identify", "Gemini discovers emerging attack hypotheses."),
    ("02", "Generate", "Hypotheses become executable attack genomes."),
    ("03", "Attack", "Red Team evolves harder synthetic trajectories."),
    ("04", "Defend", "Blue Team scores and explains each candidate."),
    ("05", "Adapt", "Autopsies feed mutations into the next generation."),
]
for col, (num, title, desc) in zip(cols, stages):
    with col:
        st.markdown(
            f'<div class="stage"><div class="muted">{num}</div>'
            f'<h3>{title}</h3><div class="small">{desc}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# ---------- Run ----------
if run:
    with st.status("Running EVO-PAY adaptive loop…", expanded=True) as status:
        status.write("Generating attacks, evaluating them against Blue Team, and feeding back autopsies.")
        code, stdout, stderr = run_closed_loop(generations, population, discover)

        if stdout:
            st.code(stdout, language="text")
        if stderr:
            with st.expander("Runtime diagnostics"):
                st.code(stderr, language="text")

        if code == 0:
            status.update(label="Closed loop completed", state="complete")
            st.session_state["last_output"] = stdout
        else:
            status.update(label=f"Closed loop exited with code {code}", state="error")
            st.session_state["last_output"] = stdout + "\n" + stderr

# ---------- Results ----------
output = st.session_state.get("last_output", "")

if output:
    generations_df, metrics, discoveries, autopsies = parse_results(output)

    st.subheader("Live experiment results")

    if not generations_df.empty:
        latest = generations_df.iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Latest detection", f"{latest.detection:.1f}%")
        k2.metric("Attack success", f"{latest.attack_success:.1f}%")
        k3.metric("Avg. risk", f"{latest.avg_risk:.4f}")
        k4.metric("Attack diversity", f"{latest.diversity:.4f}")

        st.markdown("### Red Team adaptation")
        chart = generations_df.set_index("generation")[
            ["detection", "attack_success"]
        ]
        st.line_chart(chart)

        st.markdown("### Evolution dynamics")
        chart2 = generations_df.set_index("generation")[
            ["avg_fitness", "avg_risk", "diversity"]
        ]
        st.line_chart(chart2)

        st.markdown("### Generation-by-generation scoreboard")
        display_df = generations_df.copy()
        display_df["detection"] = display_df["detection"].map(lambda x: f"{x:.1f}%")
        display_df["attack_success"] = display_df["attack_success"].map(lambda x: f"{x:.1f}%")
        display_df["avg_fitness"] = display_df["avg_fitness"].map(lambda x: f"{x:.4f}")
        display_df["avg_risk"] = display_df["avg_risk"].map(lambda x: f"{x:.4f}")
        display_df["diversity"] = display_df["diversity"].map(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    if metrics:
        st.markdown("### Blue Team performance")
        mcols = st.columns(5)
        for col, key, label in zip(
            mcols,
            ["precision", "recall", "f1", "roc_auc", "pr_auc"],
            ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        ):
            if key in metrics:
                col.metric(label, f"{metrics[key]:.4f}")

        extra = st.columns(3)
        if "fpr" in metrics:
            extra[0].metric("False-positive rate", f"{metrics['fpr']:.2%}")
        if "recall_at_1pct_fpr" in metrics:
            extra[1].metric("Recall @ 1% FPR", f"{metrics['recall_at_1pct_fpr']:.4f}")
        if "recall_at_5pct_fpr" in metrics:
            extra[2].metric("Recall @ 5% FPR", f"{metrics['recall_at_5pct_fpr']:.4f}")

    st.markdown("### GenAI contribution")
    g1, g2 = st.columns(2)
    g1.metric("Threat discoveries", discoveries if discoveries is not None else "—")
    g2.metric("Attack autopsies", autopsies if autopsies is not None else "—")

    with st.expander("Raw experiment output"):
        st.code(output, language="text")
else:
    st.info(
        "Start the Blue Team service, then click **Run Adaptive Closed Loop**. "
        "The dashboard uses the existing `integration.closed_loop` entry point, "
        "so the UI does not duplicate your fraud logic."
    )

st.caption("EVO-PAY • Synthetic environment only • No real payment data or live transactions")
