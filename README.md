# EVO-PAY

**A self-evolving Red Team / Blue Team system for adaptive payment-fraud defense**

Built for the [Mastercard Innovation Challenge 2026](https://www.kaggle.com/competitions/mastercard-innovation-challenge-2026).

EVO-PAY closes the loop between attack and defense: a GenAI-guided Red Team
discovers and evolves synthetic fraud campaigns, a Blue Team classifier
detects them, and every detection failure is analyzed and fed back as
mutation pressure for the next generation of attacks. The core question this
project answers is not "can we detect fraud?" but **"does the defense
measurably get better because of what the attacker just learned?"**

> **Core thesis:** Red Team attacks become the training and stress-testing
> ground for Blue Team, while Blue Team's failures drive the next generation
> of attacks.

---

## Table of contents

- [Why this is different](#why-this-is-different)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
  - [1. Environment setup](#1-environment-setup)
  - [2. Train the baseline Blue Team model](#2-train-the-baseline-blue-team-model)
  - [3. Start the Blue Team API](#3-start-the-blue-team-api)
  - [4. Run the closed loop](#4-run-the-closed-loop)
  - [5. Launch the dashboard](#5-launch-the-dashboard)
- [GenAI layer](#genai-layer)
- [Evaluation](#evaluation)
  - [Metrics tracked](#metrics-tracked)
  - [Held-out generalization](#held-out-generalization)
  - [Reproducibility: seed sweeps and frozen artifacts](#reproducibility-seed-sweeps-and-frozen-artifacts)
- [Sample results](#sample-results)
- [Competition evaluation mapping](#competition-evaluation-mapping)
- [Known limitations](#known-limitations)
- [Tests](#tests)
- [Security](#security)

---

## Why this is different

Most fraud-detection prototypes are a classifier trained once on a static
dataset. EVO-PAY is a closed loop:

```
GenAI discovers a threat hypothesis
        ↓
Red Team evolves it into synthetic attack campaigns
        ↓
Blue Team scores and (mis)detects them
        ↓
GenAI autopsies the failure and recommends a mutation
        ↓
Red Team mutates and tries again
        ↓
Blue Team retrains on what it just missed
        ↺
```

Every piece of that loop is instrumented so the adaptation is measurable, not
just claimed:

- **Lineage tracking** — every mutated campaign records its parent, so any
  successful evasion can be traced back through the exact sequence of changes
  that produced it.
- **Time-to-Adapt** — how many generations Blue Team needs to bring detection
  back above a target threshold after Red Team finds a gap.
- **Static vs. adaptive comparison** — the same attacks scored against a
  frozen detector snapshot and against the continuously retrained one, to
  prove degradation is being caused by Red Team evolving, not just noise.
- **Held-out evaluation** — a reserved set of attack families and primitive
  combinations that never enter the evolutionary loop, used only to test
  whether the defense generalizes beyond what it trained on.
- **Seed sweeps** — the same experiment run across multiple random seeds to
  confirm results are representative rather than a single lucky (or unlucky)
  run.

---

## Architecture

```
                    ┌──────────────────┐
                    │   Gemini GenAI   │
                    │ Threat Discovery │
                    └────────┬─────────┘
                             │
                             ▼
                    Attack Hypothesis
                             │
                             ▼
                    ┌──────────────────┐
                    │  Genome Codec    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   RED TEAM 🔴    │
                    │ Attack Evolution │
                    └────────┬─────────┘
                             │
                             ▼
                    Synthetic Campaigns
                             │
                             ▼
                    ┌──────────────────┐
                    │   BLUE TEAM 🔵   │
                    │ Fraud Detection  │
                    └────────┬─────────┘
                             │
                             ▼
                    Detection / Risk Score
                             │
                             ▼
                    ┌──────────────────┐
                    │ Attack Autopsy   │
                    │      GenAI       │
                    └────────┬─────────┘
                             │
                             ▼
                    Recommended Mutations
                             │
                             └──────────► Red Team
                                          ↺
```

GenAI never decides whether a payment is fraudulent — its output is
schema-validated, realism-gated, and used only as a hypothesis/mutation
input. Blue Team remains the sole source of truth for risk and detection
decisions.

---

## Repository layout

```
EVO-PAY/
│
├── api/                     # Blue Team FastAPI service
├── dashboard/                # Streamlit prototype dashboard
├── ui/
│   └── app.py                 # Competition-facing demo UI
│
├── data/                     # Synthetic dataset generation + raw data
├── features/                 # Behavioral / temporal / graph feature engineering
├── models/                   # Blue Team training pipeline + saved models
├── policy/                   # Decisioning / reason-code logic
│
├── genai/                    # Gemini client, threat discovery, attack autopsy
├── red_and_blue_team/
│   ├── red_team.py             # Attack genomes, evolution, campaign generation
│   ├── blue_team.py            # Trainable detector, retraining
│   └── ecosystem.py            # Synthetic payment ecosystem
│
├── integration/
│   └── closed_loop.py          # Orchestrates the full Discover→Attack→Detect→
│                                #   Autopsy→Mutate→Repeat loop
│
├── eval/
│   ├── holdout.py               # Held-out family/combo generalization testing
│   ├── lineage.py                # Attack lineage graph + visualization
│   ├── time_to_adapt.py          # Time-to-Adapt KPI
│   └── red_team_benchmark.py
│
├── experiments/
│   ├── seed_sweep.py             # Multi-seed reproducibility sweep
│   └── frozen_seed_13/           # Frozen, documented demonstration run
│
├── notebooks/                 # Exploratory analysis
├── tests/                     # pytest suite
│
├── IDENTIFY.md                 # Threat landscape + simulator coverage matrix
├── INTEGRATION.md              # Live Gemini + Blue Team integration details
├── README_CLOSED_LOOP.md       # Closed-loop deep dive
├── .env.example
└── requirements.txt
```

---

## Getting started

### 1. Environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the environment template and add your Gemini key (optional — the system
runs fully offline without it):

```powershell
copy .env.example .env
# then edit .env:
# GEMINI_API_KEY=your_key_here
```

`.env` is git-ignored. Never commit real API keys — see [Security](#security).

### 2. Train the baseline Blue Team model

```powershell
python -c "from data.generate_synthetic import generate_dataset; from pathlib import Path; Path('data/raw').mkdir(parents=True, exist_ok=True); generate_dataset(n_customers=300, n_legit_per_customer=40, fraud_ratio=0.06, seed=7).to_csv('data/raw/synthetic_transactions.csv', index=False)"
python -m models.train_pipeline
```

### 3. Start the Blue Team API

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

API docs: `http://127.0.0.1:8000/docs`

### 4. Run the closed loop

Zero-API integration test (no GenAI calls, fully deterministic):

```powershell
python -m integration.closed_loop --local-blue --no-genai --generations 2 --population 12
```

Full loop with live Gemini threat discovery and attack autopsy:

```powershell
python -m integration.closed_loop --generations 3 --population 12 --discover 4
```

Outputs are written to `integration/results/`:

| File | Contents |
|---|---|
| `strategy_memory.csv` | Red Team evaluation memory (genome, fitness, family, lineage) |
| `fraud_transactions.csv` | Generated attack campaigns |
| `generation_stats.csv` | Per-generation fitness / risk / detection / diversity stats |
| `genai_discoveries.json` | GenAI threat hypotheses, including research-only ideas |
| `genai_autopsies.json` | Post-detection explanations and mutation guidance |
| `static_vs_adaptive.json` | Detection rate against frozen vs. retrained Blue Team |
| `time_to_adapt.json` | Generations needed to recover detection per attack family |
| `closed_loop_summary.json` | Full run metadata and summary metrics |

### 5. Launch the dashboard

Competition demo UI:

```powershell
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

Development/exploration dashboard:

```powershell
python -m streamlit run dashboard/app.py
```

Both read from `integration/closed_loop`'s output rather than duplicating
Red/Blue/GenAI logic, and visualize the adaptive loop, generation metrics,
Blue Team performance, static-vs-adaptive comparison, attack lineage, and
Time-to-Adapt.

---

## GenAI layer

EVO-PAY uses Gemini as a **threat ideation and explanation layer**, never as
the transaction generator or the detector itself:

- **Threat discovery** (`genai/discover.py`) — proposes structured attack
  hypotheses, separated into transaction-simulatable and research-only ideas.
  Only simulatable hypotheses are converted into attack genomes.
- **Attack autopsy** — after each Blue Team evaluation, explains why an
  attack was or wasn't detected, names the weakest signal involved, and
  recommends specific genome mutations, which feed back into the next Red
  Team generation.

Smoke test (requires `GEMINI_API_KEY`, makes live API calls):

```powershell
python -m genai.smoke_test
```

Actual synthetic attack execution is deterministic Python logic, so
experiments remain reproducible even though the ideation step is
GenAI-driven.

---

## Evaluation

### Metrics tracked

Precision, recall, F1, false-positive rate, ROC-AUC, PR-AUC, and recall at
fixed low-FPR operating points (1% and 5%) — evaluated against both Red Team
attacks and legitimate transaction traffic.

### Held-out generalization

`eval/holdout.py` reserves specific attack families and primitive
combinations that are **excluded from the evolutionary loop entirely** and
used only for a final, independent test. This exists because in-distribution
performance is not evidence of generalization:

```
In-distribution performance   ≠   Generalization to unseen attacks
```

### Reproducibility: seed sweeps and frozen artifacts

`experiments/seed_sweep.py` runs the closed loop across multiple seeds and
scores each run for how well it demonstrates the adaptive-defense thesis
(declining static detection, stable adaptive detection, preserved attack
diversity). The winning configuration is frozen under
`experiments/frozen_seed_13/` as the documented, reproducible demonstration
run referenced in the competition write-up.

---

## Sample results

Results below are from the frozen, seeded demonstration run
(`experiments/frozen_seed_13/`) and represent one documented experimental
configuration — not a universal performance guarantee. Regenerate before
citing as current.

| Metric | Value |
|---|---|
| Precision | 1.0000 |
| Recall | 0.7742 |
| F1 | 0.8727 |
| False-positive rate | 0.0000 |
| ROC-AUC | 0.9400 |
| PR-AUC | 0.9705 |

| Held-out evaluation | Result |
|---|---|
| Reserved campaigns tested | 20 |
| Detected | 65.0% |
| Legitimate-baseline FPR | 0.0% |

*(See `experiments/frozen_seed_13/closed_loop_summary.json` and
`holdout_eval.json` for the full, current numbers.)*

---

## Competition evaluation mapping

| Criterion | How EVO-PAY addresses it |
|---|---|
| **Diversity of attacks** | GenAI threat discovery, a 7-dimensional attack genome, family-balanced selection, and seed sweeps to prevent evolutionary collapse into a single dominant strategy |
| **Fidelity of attacks** | A modeled synthetic payment ecosystem (customers, accounts, merchants) with behavioral, temporal, and graph-derived features, not abstract feature vectors |
| **Detection algorithms** | Blue Team classifier with the full standard metric suite, evaluated both in-distribution and on held-out, never-trained-on attack combinations |
| **Novelty** | A closed adversarial loop where Blue Team's failures directly drive Red Team's next generation, made legible via lineage tracking and the Time-to-Adapt KPI |
| **Real-world feasibility** | Modular architecture (discovery / generation / detection / evaluation are independently swappable), inline-scoring design with no GenAI dependency on the critical detection path, and reason codes for auditability |

---

## Known limitations

Read before presenting results as more than what they are:

- This is a synthetic adversarial simulation, not a production Mastercard
  fraud-detection system. Synthetic transaction data alone cannot establish
  real-world fraud performance.
- Red Team and Blue Team still operate over somewhat different synthetic-
  world distributions; cross-domain calibration should be validated before
  treating risk scores as production-grade probabilities.
- The graph layer currently uses engineered graph features rather than a
  learned, global cross-customer fraud-ring GNN.
- GenAI hypotheses are useful for expanding the explored attack surface but
  require domain-expert validation before informing any operational
  decision.
- Local latency measurements are an engineering signal, not a production
  payment-network latency guarantee.
- Seed-specific results should not be presented as universal — see the seed
  sweep in `experiments/`.
- Blue Team retraining is a deliberate, separate deployment step; a running
  API process caches model artifacts and must be explicitly reloaded before
  a newly retrained model is "live."

---

## Tests

```powershell
pytest -q
```

Key suites: `tests/test_red_blue_closed_loop.py`, `tests/test_holdout.py`.
Re-run after any change to core integration or evaluation logic.

Sanity-check compilation across the repo:

```powershell
python -m compileall -q .
```

---

## Security

EVO-PAY is a synthetic research environment and does not connect to real
payment rails. Never commit `.env`, production credentials, cardholder data,
or any other sensitive payment information to this repository. Verify before
every push:

```powershell
git status
```

`.env` must never appear as staged or untracked.
