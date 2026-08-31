# EVO-PAY Demo Script

> **For the presenter.** This is a scripted walkthrough of the key "aha" moments
> in the demo. Follow these steps in order for maximum impact.

## Setup (before the demo)

1. Ensure Blue Team API is running:
   ```powershell
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```

2. Run the frozen demo (do this ahead of time to avoid API flakiness):
   ```powershell
   python -m integration.closed_loop --local-blue --no-genai --profile demo --seed 7
   ```

3. Open both dashboards:
   ```powershell
   streamlit run dashboard/app.py
   streamlit run ui/app.py
   ```

---

## Demo Path (5-7 minutes)

### Act 1: "The Problem" (1 min)

> "Traditional fraud detection is static — you train a model and deploy it.
> But attackers evolve. Here's what happens when we freeze a defense and
> let attacks evolve against it."

**Show**: The two-curve chart in the "Adaptive Closed Loop" page.
- Point to the **Static Defense** line (orange) declining from ~33% → ~11%.
- "This is what happens to a fixed defense over time."

### Act 2: "The Solution" (1 min)

> "EVO-PAY closes the loop. After each generation of attacks, the Blue Team
> retrains on what it just saw."

**Show**: The **Adaptive Defense** line (blue) staying flat or recovering.
- "Same attacks, same generations — but the adaptive detector keeps up."
- "This is the core thesis: continuous adversarial retraining."

### Act 3: "How It Works — The Autopsy" (2 min)

> "Here's where GenAI adds value. After each round, Gemini performs an
> autopsy on the attacks Blue missed."

**Show**: Navigate to the "GenAI Autopsy → Mutation Trace" section.
1. Click on an autopsy entry where `detected = False` (attack Blue missed).
2. **Show the three columns**:
   - 🔴 Attack: "This attack used [dimensions] to evade detection."
   - 🔵 Defense Response: "Blue only flagged [reason codes] — it missed [weakest signal]."
   - 🧬 GenAI Autopsy: "Gemini explains *why* it was missed and recommends mutations."

3. **Show the Genome Mutation Diff**:
   - "Here's the specific mutation Gemini recommended: increase [dimension] by 0.10."
   - "In the next generation, Red used this mutation — and it succeeded/was now caught."

### Act 4: "Diversity of Attacks" (1 min)

> "The evolutionary search doesn't just find one trick — it explores the
> entire attack surface."

**Show**: The generation-by-generation scoreboard.
- Point to the "families" column showing multiple attack families per generation.
- Point to the "diversity" column showing maintained genome diversity.
- "Family-balanced selection prevents any single attack type from dominating."

### Act 5: "Real-World Feasibility" (1 min)

> "This isn't just an academic exercise."

**Show**: The latency metrics.
- "Inference takes ~45ms per campaign at p50 — well within payment authorization SLAs."
- "The false-positive rate of X% translates to Y flagged legitimate transactions per
  million — manageable with step-up authentication."

**Mention**: "Full feasibility analysis is in the write-up, including an honest
acknowledgment that these results are on synthetic data."

### Closing (30 sec)

> "EVO-PAY demonstrates a methodology for continuous adversarial defense.
> The key insight isn't any single detection — it's that the defense
> adapts as fast as the attacks evolve."

---

## If Things Go Wrong

- **Gemini 503**: Use `--no-genai` to skip GenAI calls. The evolutionary loop
  still works without GenAI; you just lose the autopsy/discovery features.
- **API unreachable**: Use `--local-blue` for the trainable local detector.
- **Slow run**: Use `--profile quick` for a 3-generation, 12-population demo.
- **Pre-cached results**: If all else fails, the frozen run in
  `experiments/demo_run_final.json` has all the data the charts need.
