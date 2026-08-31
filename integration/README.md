# EVO-PAY Integration

The integration layer is the boundary between the Red and Blue teams.

## Closed loop

1. GenAI proposes structured threat hypotheses.
2. Valid hypotheses become `AttackGenome` seeds.
3. Red evolves and simulates campaigns.
4. Blue evaluates campaigns through the real API.
5. GenAI performs a post-decision attack autopsy.
6. Bounded mutation guidance seeds the next Red generation.
7. Optionally, Red-generated fraud is appended to Blue's training data and the canonical Blue training pipeline is run.
8. Results are persisted for the dashboard/write-up.

GenAI never supplies the fraud decision and never bypasses Red realism
validation or Blue scoring.

## Run the wiring loop without GenAI

With Blue running on port 8000:

```bash
python -m integration.closed_loop --no-genai --generations 2 --population 12
```

## Run the full GenAI loop

Set `OPENAI_API_KEY`, start Blue, then:

```bash
python -m integration.closed_loop --generations 3 --population 12 --discover 4
```

Blue retraining is intentionally a separate deployment step in this version.
The closed-loop orchestrator must not claim that a newly written model is live
while the running API still holds the previous model in memory.

Outputs are written to `integration/results/`.
