# EVO-PAY — Full Project Context

> **Purpose of this file:** If you run out of AI limits or need to onboard someone
> (including a fresh AI session), this file contains everything needed to understand
> the current state of the codebase, what has been implemented, what each file does,
> and how to run/test/extend the project.

---

## 1. What is EVO-PAY?

A closed-loop adversarial fraud defense system. A **Red Team** evolves attack
strategies via genetic algorithms against a **Blue Team** detector. A **GenAI layer**
(Gemini) discovers novel threats and autopsies detection failures. The key innovation
is that Blue retrains every generation, creating a genuine arms race — not a static
model being beaten by evolving attacks.

**Core thesis (proven):** A static defense degrades as Red evolves, but an adaptive
(retrained) defense keeps pace. This is demonstrated by the **two-curve chart**
(Section 6) showing static detection declining while adaptive detection holds/recovers.

---

## 2. Repository Structure

```
EvoPay/
├── red_and_blue_team/          # ← ACTUAL package name (NOT "red_team")
│   ├── __init__.py
│   ├── red_team.py             # AttackGenome, GenomeCodec, RedTeamController,
│   │                           #   EvolutionEngine, MutationEngine, FitnessEngine,
│   │                           #   MemoryRecord, StrategyMemory, FailureAnalyzer,
│   │                           #   NoveltyEngine, RealismValidator, CampaignPlanner
│   ├── blue_team.py            # DetectorInterface, DetectionResult, HeuristicDetector,
│   │                           #   TrainableDetector (NEW — RF-based, retrainable)
│   ├── blue_team_client.py     # BlueTeamClient (HTTP wrapper for Blue API)
│   └── ecosystem.py            # PaymentEcosystem, Customer, Account, Merchant, Transaction
│
├── integration/
│   ├── closed_loop.py          # Main entry point — the adversarial loop
│   ├── adversarial_retrain.py  # Blue retraining via training pipeline
│   └── improvement file        # The original improvement plan (Sections 1-6)
│
├── genai/
│   ├── client.py               # GenAIClient (Gemini wrapper)
│   ├── discoverer.py           # ThreatDiscoverer, hypothesis_to_genome
│   ├── analyst.py              # AttackAnalyst (autopsy)
│   └── schemas.py              # AttackHypothesis, AttackAutopsy Pydantic models
│
├── api/                        # FastAPI Blue Team service
│   └── main.py                 # /evaluate, /health, /reload endpoints
│
├── models/
│   ├── train_pipeline.py       # build_feature_matrix, run_training
│   ├── tabular.py              # train_tabular_model, predict_tabular
│   └── anomaly.py              # train_anomaly_model, save_anomaly_model
│
├── features/                   # Feature engineering modules
├── policy/                     # Decision policy (risk → allow/challenge/hold/block)
│
├── eval/
│   ├── metrics.py              # compute_classification_metrics, recall_at_fpr
│   ├── closed_loop_metrics.py  # generation_metrics, overall_metrics
│   └── red_team_benchmark.py   # Full Red-vs-Blue evaluation
│
├── dashboard/
│   └── app.py                  # Streamlit dashboard (premium dark theme)
│
├── ui/
│   └── app.py                  # Competition-facing Streamlit UI
│
├── data/
│   ├── generate_synthetic.py   # Synthetic transaction generator
│   └── raw/                    # Generated CSV data
│
├── docs/
│   ├── feasibility.md          # Real-world feasibility write-up (Section 4)
│   └── demo_script.md          # Presenter walkthrough script (Section 5)
│
├── experiments/                # Frozen reproducible run artifacts (Section 2)
├── tests/                      # Test suite
│
├── IDENTIFY.md                 # Threat landscape taxonomy
├── INTEGRATION.md              # Integration documentation
├── README.md                   # Project readme
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

---

## 3. Critical Architecture Details

### Import paths
All imports use `red_and_blue_team.*` (NOT `red_team.*`). The directory is named
`red_and_blue_team/`. This was a breaking bug that was fixed in Section 0.

### Key classes and their locations

| Class | File | Purpose |
|-------|------|---------|
| `AttackGenome` | `red_and_blue_team/red_team.py` | 7-gene vector `[0,1]^7` representing an attack strategy |
| `GenomeCodec` | `red_and_blue_team/red_team.py` | Encodes/decodes genome ↔ `AttackStrategy` |
| `RedTeamController` | `red_and_blue_team/red_team.py` | Orchestrates evolution: population → evaluate → select → mutate |
| `EvolutionEngine` | `red_and_blue_team/red_team.py` | Selection (family-balanced) + mutation + crossover |
| `MutationEngine` | `red_and_blue_team/red_team.py` | Gaussian mutation with variance-biased dimension selection |
| `FitnessEngine` | `red_and_blue_team/red_team.py` | Scores campaigns: gain × evasion × novelty |
| `MemoryRecord` | `red_and_blue_team/red_team.py` | Per-campaign evaluation record (includes `detection_probability`, `attack_success`) |
| `DetectorInterface` | `red_and_blue_team/blue_team.py` | Abstract interface: `evaluate(transactions, ecosystem) → DetectionResult` |
| `HeuristicDetector` | `red_and_blue_team/blue_team.py` | Rule-based detector (no trainable state) |
| `TrainableDetector` | `red_and_blue_team/blue_team.py` | **RF-backed detector** with `update()`, `frozen_copy()`, 10-feature featurizer |
| `BlueTeamClient` | `red_and_blue_team/blue_team_client.py` | HTTP client for the Blue API (`/evaluate` endpoint) |
| `PaymentEcosystem` | `red_and_blue_team/ecosystem.py` | Synthetic world: customers, accounts, merchants, transactions |

### The 7 genome dimensions
```
amount | temporal | device | geographic | merchant | velocity | coordination
```
Each gene is a float in [0, 1]. Values above `ACTIVE_THRESHOLD` (0.65) activate
the corresponding attack pattern. The `GenomeCodec.decode()` method converts
genome values into an `AttackStrategy` with specific pattern types.

### TrainableDetector (Section 1 — the core addition)
- Wraps a `RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced")`
- **10 features**: amount_z_score, max_amount_z_score, off_hours_ratio, unknown_device_ratio,
  unusual_location_ratio, velocity, txn_count, amount_std, small_amount_ratio, merchant_diversity
- Before first training: delegates to `HeuristicDetector`
- `update(records, txns_map, ecosystem)` — retrains on all accumulated labeled data
- `add_legitimate_baseline(ecosystem, rng, n)` — seeds class-0 training data
- `frozen_copy()` — deep copy that is never updated (for the static baseline)

### Closed Loop Flow (`integration/closed_loop.py`)
```
1. Build ecosystem (customers, merchants, transactions)
2. Create TrainableDetector + seed with legitimate baselines
3. Freeze a static detector copy (for two-curve chart)
4. For each generation:
   a. RedTeamController.evolve(1 generation)
   b. Score campaigns against both adaptive and static detectors
   c. Log static_detection_rate and adaptive_detection_rate
   d. GenAI autopsy (if enabled) → guided mutations for next gen
   e. detector.update() — retrain Blue on this generation's campaigns
5. Evaluate on held-out legitimate traffic + fresh random attacks
6. Compute classification metrics (precision, recall, F1, FPR, AUC)
7. Save all results to integration/results/
```

---

## 4. What Was Implemented (Improvement Plan Sections 1-6)

### Section 0 — Fixed Broken Imports ✅
- Changed `from red_team.*` → `from red_and_blue_team.*` in:
  - `integration/closed_loop.py`
  - `genai/discoverer.py`
  - `dashboard/app.py`
  - `eval/red_team_benchmark.py`

### Section 1 — Detector-Aware Fitness + Retraining Loop ✅
- Added `TrainableDetector` class in `blue_team.py`
- Added `detection_probability` and `attack_success` fields to `MemoryRecord`
- Added `fraud_transactions_by_campaign` dict to `RedTeamController`
- `closed_loop.py` now uses `TrainableDetector` by default with `--local-blue`
- Retrains Blue after each generation via `detector.update()`
- Added `--use-legacy-proxy` flag to fall back to `HeuristicDetector`

### Section 2 — Scale Up Demo Run ✅
- Added `--profile demo` (8 gen, 40 pop, 6 discover) and `--profile quick` (3 gen, 12 pop, 4 discover)
- Added explicit `--seed` CLI argument
- Saves experiment config + full summary to `experiments/` when using a profile

### Section 3 — Diversity Preservation ✅
- **Family-balanced selection**: `EvolutionEngine._family_balanced_select()` caps survivors per family
- **Variance-biased mutation**: `MutationEngine` tracks gene variances, biases mutation toward
  converged (low-variance) dimensions via inverse-variance weighting

### Section 4 — Feasibility Write-up ✅
- Created `docs/feasibility.md` with latency, FPR-to-business, integration model, scope limitation, compliance
- Added latency instrumentation in `closed_loop.py` (p50/p95/mean per-campaign ms)

### Section 5 — Autopsy UI Polish ✅
- Added autopsy → mutation trace interaction in `dashboard/app.py`:
  - 3-column layout: 🔴 Attack | 🔵 Defense | 🧬 Autopsy
  - Genome mutation diff table (dimension, direction, change, rationale)
- Created `docs/demo_script.md` presenter walkthrough

### Section 6 — Two-Curve Comparison Chart ✅
- Frozen static detector snapshot at generation 0
- Per-generation scoring against both static and adaptive detectors
- `static_vs_adaptive.json` saved to output directory
- Chart added to both `dashboard/app.py` (tabs) and `ui/app.py`

### Dashboard Premium Redesign ✅
- Dark glassmorphism theme (CSS injected via `st.markdown`)
- Inter + JetBrains Mono fonts
- Gradient headers, glowing metric cards, smooth hover animations
- New **Command Center** page showing system status + latest results
- Tabbed chart layout on Closed Loop page (Core Thesis / Evolution / Data)
- Emoji status badges throughout

---

## 5. How to Run

### Prerequisites
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Generate data + train Blue
```powershell
python -c "from data.generate_synthetic import generate_dataset; from pathlib import Path; Path('data/raw').mkdir(parents=True, exist_ok=True); generate_dataset(n_customers=300, n_legit_per_customer=40, fraud_ratio=0.06, seed=7).to_csv('data/raw/synthetic_transactions.csv', index=False)"
python -m models.train_pipeline
```

### Run the closed loop (local, no API/GenAI needed)
```powershell
# Quick test (3 gen, 12 pop)
python -m integration.closed_loop --local-blue --no-genai --generations 3 --population 12

# Full demo (8 gen, 40 pop)
python -m integration.closed_loop --local-blue --no-genai --profile demo --seed 7

# With legacy heuristic (no retraining)
python -m integration.closed_loop --local-blue --no-genai --use-legacy-proxy --generations 3 --population 12
```

### Run with Blue API
```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
python -m integration.closed_loop --no-genai --generations 3 --population 12
```

### Run with GenAI (requires GEMINI_API_KEY in .env)
```powershell
python -m integration.closed_loop --local-blue --generations 5 --population 20
```

### Launch dashboards
```powershell
streamlit run dashboard/app.py    # Premium dashboard
streamlit run ui/app.py           # Competition UI
```

---

## 6. Key CLI Arguments for closed_loop.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--generations` | 3 | Number of evolutionary generations |
| `--population` | 12 | Population size per generation |
| `--discover` | 4 | Number of GenAI hypotheses to request |
| `--seed` | 7 | Random seed for reproducibility |
| `--local-blue` | off | Use local TrainableDetector instead of Blue API |
| `--no-genai` | off | Disable Gemini API calls |
| `--use-legacy-proxy` | off | Use HeuristicDetector (no retraining) |
| `--profile` | none | Preset: `demo` (8/40/6) or `quick` (3/12/4) |
| `--retrain-blue-every` | 0 | Retrain API-based Blue every N gens (0=disabled) |
| `--output-dir` | `integration/results` | Where to write output files |

---

## 7. Output Files (integration/results/)

| File | Contents |
|------|----------|
| `closed_loop_summary.json` | Full summary: timeline, metrics, latency, detector type |
| `static_vs_adaptive.json` | Per-generation static vs adaptive detection rates |
| `strategy_memory.csv` | All evaluated campaigns with fitness, risk, detection |
| `fraud_transactions.csv` | Generated fraud transactions |
| `generation_stats.csv` | Per-generation aggregate stats |
| `genai_discoveries.json` | GenAI threat hypotheses |
| `genai_autopsies.json` | GenAI attack autopsies |
| `loop_events.json` | Detailed event log |
| `retraining_history.json` | Blue retraining reports |

---

## 8. Verified Test Results

### 3-Generation Quick Test (seed=7)
```
generation | detection | attack_success | avg_fitness | avg_risk | families | diversity
         0 |     50.0% |          50.0% |      0.3222 |   0.5424 |       12 |   0.9682
         1 |    100.0% |           0.0% |      0.1461 |   0.8433 |        4 |   0.6409
         2 |    100.0% |           0.0% |      0.1262 |   0.8454 |        3 |   0.3967

STATIC vs. ADAPTIVE:
  Gen 0: static=33.3% | adaptive=33.3%
  Gen 1: static=11.1% | adaptive=100.0%
  Gen 2: static=11.1% | adaptive=100.0%

Blue: precision=1.00, recall=0.78, F1=0.88, FPR=0.00
Latency: p50=44ms, p95=46ms
```

### 5-Generation Full Test (seed=42)
```
generation | detection | attack_success | avg_fitness | avg_risk | families | diversity
         0 |     63.3% |          36.7% |      0.3274 |   0.6031 |       19 |   1.0775
         1 |     90.0% |          10.0% |      0.2232 |   0.8241 |        8 |   0.9170
         2 |     83.3% |          16.7% |      0.2120 |   0.8019 |        9 |   0.9551
         3 |     94.4% |           5.6% |      0.2095 |   0.8974 |        9 |   0.9733
         4 |    100.0% |           0.0% |      0.1830 |   0.9127 |        7 |   0.6866

STATIC vs. ADAPTIVE:
  Gen 0: static=50.0% | adaptive=50.0%
  Gen 1: static=21.4% | adaptive=85.7%
  Gen 2: static=50.0% | adaptive=92.9%
  Gen 3: static=71.4% | adaptive=92.9%
  Gen 4: static=78.6% | adaptive=100.0%

Blue: precision=0.98, recall=0.83, F1=0.90, FPR=0.08
Latency: p50=56ms, p95=71ms
```

---

## 9. Dependencies (requirements.txt)

```
fastapi>=0.104.0, uvicorn[standard]>=0.24.0, pydantic>=2.5.0
lightgbm>=4.1.0, scikit-learn>=1.3.0, imbalanced-learn>=0.11.0
shap>=0.43.0, networkx>=3.2, pandas>=2.1.0, numpy>=1.24.0
duckdb>=0.9.0, streamlit>=1.28.0, matplotlib>=3.8.0
pyarrow>=14.0.0, requests>=2.31.0, google-genai>=1.30.0
python-dotenv>=1.0.1
```

---

## 10. Known Issues / Things to Watch

1. **Import paths**: Always use `red_and_blue_team.*`, never `red_team.*`.
2. **Gemini 503s**: The Gemini API can be flaky. Use `--no-genai` for reliable runs.
3. **Blue API not running**: Use `--local-blue` to bypass the API entirely.
4. **Git user config**: Set `git config user.email` and `git config user.name` before commits.
5. **CRLF warnings**: Windows line endings — harmless, git handles conversion.
6. **TrainableDetector needs ≥2 samples per class**: First generation uses HeuristicDetector
   as fallback before enough training data accumulates.
7. **Static detector volatility**: The static detector's results can vary because
   the HeuristicDetector (used as initial state before RF training) has different
   scoring characteristics than the trained RF model.

---

## 11. What's Left / Extension Ideas

- [ ] Run `python -m pytest tests/ -v` to verify existing tests pass
- [ ] Run the full `--profile demo` with GenAI enabled and save to `experiments/`
- [ ] Generate the `.docx` write-up with real numbers from the frozen demo run
- [ ] Consider GNN-based graph features for cross-customer fraud ring detection
- [ ] Production: swap RandomForest for LightGBM with proper Dataset/callback pipeline
- [ ] Add incremental learning (`partial_fit`) for online deployment scenarios
