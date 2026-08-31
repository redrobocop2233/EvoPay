# EVO-PAY UI

A presentation layer for the existing EVO-PAY adaptive closed-loop system.

## Run

From the EVO-PAY repository root:

```powershell
pip install -r ui/requirements.txt
streamlit run ui/app.py
```

The UI expects the existing project entry point:

```powershell
python -m integration.closed_loop --generations 3 --population 12 --discover 4
```

It also checks the existing Blue Team health endpoint:

```text
http://127.0.0.1:8000/health
```

## What it shows

- Identify → Generate → Attack → Defend → Adapt architecture
- Closed-loop experiment controls
- Generation-by-generation detection and attack-success curves
- Fitness, risk and diversity evolution
- Blue Team precision, recall, F1, ROC-AUC and PR-AUC
- False-positive rate and recall at low FPR
- GenAI discovery/autopsy counts
- Raw experiment output for debugging/demo purposes

## Important

This UI deliberately calls the existing `integration.closed_loop` entry point instead of reimplementing Red Team or Blue Team logic. That keeps the demo aligned with the tested system.
