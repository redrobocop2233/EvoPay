# EVO-PAY

EVO-PAY is a closed-loop synthetic payment-fraud red/blue system. A deterministic payment simulator provides high-fidelity transaction trajectories; a genome-driven Red Team searches those trajectories; the Blue Team scores them through a FastAPI ML pipeline; and an optional GenAI layer discovers structured threat hypotheses and performs post-detection attack autopsies.

## Architecture

```text
GenAI threat discovery
        ↓
structured AttackHypothesis
        ↓
AttackGenome
        ↓
Red evolution + campaign simulation
        ↓
Blue API: behavioral + temporal + tabular + anomaly + graph + policy
        ↓
risk / decision / reason codes
        ↓
GenAI attack autopsy
        ↓
bounded mutation guidance
        ↓
next Red generation
```

GenAI never decides whether a payment is fraudulent. Its output is schema-validated, realism-gated, and used only as a hypothesis/mutation input. Blue remains the source of truth for risk and decision.

## Quick start

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt
```

Generate Blue training data and train the baseline model once:

```bash
python -c "from data.generate_synthetic import generate_dataset; from pathlib import Path; Path('data/raw').mkdir(parents=True, exist_ok=True); generate_dataset(n_customers=300, n_legit_per_customer=40, fraud_ratio=0.06, seed=7).to_csv('data/raw/synthetic_transactions.csv', index=False)"
python -m models.train_pipeline
```

Start Blue:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In another terminal, start the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

Open `http://127.0.0.1:8000/docs` to inspect the Blue API.

## Closed-loop prototype

First verify the wiring without an LLM:

```bash
python -m integration.closed_loop --no-genai --generations 2 --population 12
```

For the full GenAI loop, set `GEMINI_API_KEY` in the environment and run:

```bash
python -m integration.closed_loop --generations 3 --population 12 --discover 4
```

Outputs are written to `integration/results/`:

- `strategy_memory.csv` — Red evaluation memory
- `fraud_transactions.csv` — generated campaigns
- `generation_stats.csv` — generation-level fitness/risk/detection
- `genai_discoveries.json` — GenAI hypotheses, including research-only ideas
- `genai_autopsies.json` — post-Blue explanations and mutation guidance
- `closed_loop_summary.json` — run metadata and generation statistics

## GenAI smoke test

```bash
python -m genai.smoke_test
```

This requires `GEMINI_API_KEY` and makes live Gemini API calls.

## Repository layout

- `red_team/` — synthetic payment ecosystem, attack genomes, campaign planner, evolutionary search, local detector, and Blue API client.
- `genai/` — strict GenAI schemas, threat discovery, attack autopsy, prompts, and Gemini client.
- `api/`, `features/`, `models/`, `policy/` — Blue Team service and detection pipeline.
- `integration/` — cross-team closed-loop orchestration.
- `eval/` — Defend-pillar metrics and Red-vs-Blue benchmarks.
- `dashboard/` — Streamlit prototype.
- `IDENTIFY.md` — threat landscape and honest simulator coverage matrix.

## Important limitations

- Red and Blue still have different synthetic-world distributions; cross-domain calibration must be validated before interpreting absolute risk scores as production probabilities.
- The graph layer currently uses engineered graph features; a global cross-customer fraud-ring graph is not yet a learned GNN system.
- GenAI discovery is optional and requires an API key; deterministic Red evolution remains fully runnable without it.
- Blue retraining is intentionally a separate deployment step. A running API process caches model artifacts, so a new model must be explicitly reloaded/restarted before it can be called "live".


## Adaptive GenAI Closed Loop

The competition-facing loop is implemented in `integration/closed_loop.py`:

`Gemini discovery → AttackGenome → Red evolution → Blue detection → Gemini autopsy → bounded mutation → optional Blue retraining`.

Run `python -m integration.closed_loop --local-blue --no-genai` for a zero-API integration test, or see `INTEGRATION.md` for the live Gemini + Blue workflow.

## Competition Demo UI

The competition-facing dashboard is in `ui/`.

```powershell
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

The dashboard calls the existing `integration.closed_loop` entry point rather than duplicating Red/Blue/GenAI logic. It visualizes the adaptive loop, generation metrics, Blue Team performance, and GenAI activity.

## Clean demo sequence

1. If `models/saved/` is empty, generate training data and run `python -m models.train_pipeline` once.
2. Start Blue: `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`.
3. Verify Gemini: `python -m genai.smoke_test`.
4. Launch UI: `streamlit run ui/app.py`.
5. Run a 3-generation experiment with population 12 and 4 GenAI discoveries.

## Security

EVO-PAY is a synthetic research environment. It does not connect to real payment rails. Never place production credentials, cardholder data, or other sensitive payment information in the repository.

## Evaluation and Reproducibility

The repository includes held-out attack evaluation, evolutionary lineage tracking, a
Time-to-Adapt KPI, and deterministic seed sweeps. Quick verification:

```powershell
python -m integration.closed_loop --generations 3 --population 12 --discover 0 --local-blue --no-genai
pytest -q
python -m experiments.seed_sweep --seeds 1 7 13 --profile quick
```

Held-out attacks are reserved from the evolutionary loop and reported separately from
in-distribution performance.
