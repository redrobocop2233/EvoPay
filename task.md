# Held-Out Eval + Seed Sweep Tasks

## Section A — Held-Out Evaluation
- [x] Create `eval/holdout.py` (HoldoutConfig, violates_holdout, construct_genome, build_eval_set)
- [x] Update `red_team.py` — add holdout filtering to EvolutionEngine
- [x] Update `closed_loop.py` — replace `_holdout_attacks` with proper held-out eval
- [x] Update `docs/feasibility.md` — add in-distribution vs held-out comparison

## Section B — Reduce Saturation + Seed Sweep
- [x] Reduce `TrainableDetector` max_depth 8→5
- [x] Add adaptive mutation rate to MutationEngine
- [x] Create `experiments/seed_sweep.py`
- [x] Run verification test
- [x] Run seed sweep

## Final
- [x] Update `EVO-PAY-context.md`
- [x] Commit and push
