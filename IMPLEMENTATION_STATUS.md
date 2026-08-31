# EVO-PAY Implementation Status

## Completed

- [x] Held-out primitive-combination evaluation
- [x] Holdout filtering during evolutionary generations
- [x] Separate held-out metrics
- [x] Blue saturation mitigation (`RandomForestClassifier(max_depth=5)`)
- [x] Adaptive Red mutation pressure
- [x] Seed sweep runner
- [x] Frozen seed-13 verification run
- [x] Parent campaign lineage
- [x] Mutation summaries
- [x] Lineage evaluation module
- [x] Time-to-Adapt KPI
- [x] Dashboard lineage tab
- [x] UI headline metrics for held-out detection and Time-to-Adapt
- [x] Automated tests

## Verification

```text
compileall: PASS
pytest: 3 passed
closed-loop quick run: PASS
seed sweep (1, 7, 13): PASS
winner: seed 13
```

## Frozen verification metrics

```text
Precision       1.0000
Recall          0.7742
F1              0.8727
FPR             0.0000
ROC-AUC         0.9400
PR-AUC          0.9705
p50 latency     22.5 ms
p95 latency     28.9 ms
Held-out recall 65.0%
Held-out FPR    0.0%
```

The frozen run is a synthetic verification artifact. It must not be described as proof of
production fraud-detection accuracy.
