# EVO-PAY Adaptive Red/Blue/GenAI Integration

## The closed loop

```text
Gemini Threat Discovery
        |
        v
validated AttackHypothesis
        |
        v
AttackGenome compiler
        |
        v
Red evolutionary search -----> Blue detector
        ^                           |
        |                           v
        +---- bounded mutation <- Gemini Autopsy
                                    |
                         optional adversarial retraining
```

GenAI is deliberately **not** the transaction generator and is never the
source of truth for Blue risk. It proposes hypotheses and interprets completed
Blue decisions. Red's deterministic simulator produces the actual transactions;
Pydantic validation and the realism gate sit between GenAI and Red.

## Quick start

### 1. Install

```powershell
pip install -r requirements.txt
```

### 2. Configure Gemini

Create `.env` in the repository root:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.7-flash
```

`.env` is ignored by Git. Never commit the key.

### 3. Verify GenAI

```powershell
python -m genai.smoke_test
```

### 4. Run the complete loop locally

This uses the transparent local heuristic Blue detector and costs no API calls
when `--no-genai` is supplied:

```powershell
python -m integration.closed_loop --local-blue --no-genai --generations 3 --population 12
```

### 5. Run against the real Blue API + Gemini

Start Blue first:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then:

```powershell
python -m integration.closed_loop --generations 3 --population 12 --discover 4
```

### 6. Optional adversarial retraining

Retraining is explicit because it changes the detector. For example:

```powershell
python -m integration.closed_loop --generations 5 --population 16 --retrain-blue-every 2
```

A compatible local Blue API reloads the newly saved model through `/reload`.
If the external Blue repository does not implement `/reload`, the feed and model
artifacts are still written for that team to consume.

## Outputs

`integration/results/` contains:

- `strategy_memory.csv` — every evaluated Red campaign
- `fraud_transactions.csv` — generated synthetic attack transactions
- `generation_stats.csv` — Red search progress
- `genai_discoveries.json` — initial and adaptive hypotheses
- `genai_autopsies.json` — Blue failure explanations and mutation advice
- `loop_events.json` — chronological integration events
- `retraining_history.json` — explicit Blue retraining events
- `closed_loop_summary.json` — competition-facing metrics
- `adversarial_training_feed.csv` — base + Red data when retraining is enabled

## Metrics

The loop reports:

- Blue detection rate / recall
- precision, F1, ROC-AUC and PR-AUC
- false-positive rate on untouched legitimate transactions
- recall at 1% and 5% FPR budgets
- Red attack-success rate (`1 - campaign detection rate`)
- mean Red fitness and risk
- attack-family and behavioral-signature diversity
- genome-space diversity
- a clearly labelled simulator-fidelity proxy
- a fresh holdout attack evaluation that receives no GenAI autopsy feedback

Do not present the fidelity proxy as proof that synthetic transactions match
real-world fraud distributions. For the final write-up, validate fidelity against
an external dataset if one is available.
