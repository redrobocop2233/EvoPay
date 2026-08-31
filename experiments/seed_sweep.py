"""
Runs the closed loop across multiple seeds at demo scale, scores each run for
how well it supports the "static degrades, adaptive holds" narrative and how
well diversity is preserved, and ranks them so you can pick the best seed for
the frozen submission run.

Usage:
    python -m experiments.seed_sweep --seeds 1 7 13 21 42 99 123 --profile demo
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_one_seed(seed: int, profile: str, base_output_dir: Path) -> Path:
    out_dir = base_output_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "integration.closed_loop",
        "--local-blue", "--no-genai",
        "--profile", profile,
        "--seed", str(seed),
        "--output-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[seed {seed}] FAILED:\n{result.stderr[-2000:]}", file=sys.stderr)
    return out_dir


def score_run(out_dir: Path) -> dict | None:
    static_path = out_dir / "static_vs_adaptive.json"
    summary_path = out_dir / "closed_loop_summary.json"
    if not static_path.exists() or not summary_path.exists():
        return None

    static_data = json.loads(static_path.read_text())
    summary = json.loads(summary_path.read_text())
    
    static_rates = [row["static_detection_rate"] for row in static_data]
    adaptive_rates = [row["adaptive_detection_rate"] for row in static_data]

    # Monotonicity score: fraction of consecutive steps where static declines
    # (or stays flat) rather than rising. 1.0 = perfectly declining.
    declines = sum(
        1 for a, b in zip(static_rates, static_rates[1:]) if b <= a
    )
    monotonicity = declines / max(1, len(static_rates) - 1)

    # Total drop from first to last generation (bigger = more dramatic, good)
    static_drop = static_rates[0] - static_rates[-1]

    # Adaptive should stay high/stable — penalize big swings or early collapse
    adaptive_stability = 1.0 - float(np.std(adaptive_rates))

    timeline = summary.get("generation_timeline", [])
    if not timeline:
        return None

    diversity_start = timeline[0]["genome_diversity"]
    diversity_end = timeline[-1]["genome_diversity"]
    diversity_retention = diversity_end / max(diversity_start, 1e-6)

    families_start = timeline[0]["unique_attack_families"]
    families_end = timeline[-1]["unique_attack_families"]
    family_retention = families_end / max(families_start, 1)

    # Penalize runs where adaptive detection saturates at 100% for more than
    # one generation in a row (kills exploration pressure)
    saturated_streak = 0
    max_saturated_streak = 0
    for rate in adaptive_rates:
        if rate >= 0.999:
            saturated_streak += 1
            max_saturated_streak = max(max_saturated_streak, saturated_streak)
        else:
            saturated_streak = 0

    # Composite score — weights are a starting point, adjust after eyeballing
    # a few runs
    composite = (
        0.30 * monotonicity +
        0.20 * min(static_drop, 1.0) +
        0.20 * diversity_retention +
        0.15 * family_retention +
        0.15 * max(0.0, 1.0 - 0.25 * max_saturated_streak)
    )

    return {
        "monotonicity": round(monotonicity, 3),
        "static_drop": round(static_drop, 3),
        "diversity_retention": round(diversity_retention, 3),
        "family_retention": round(family_retention, 3),
        "max_saturated_streak": max_saturated_streak,
        "composite_score": round(composite, 4),
        "static_rates": static_rates,
        "adaptive_rates": adaptive_rates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--profile", default="demo")
    parser.add_argument("--output-dir", default="experiments/seed_sweep")
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    rows = []
    for seed in args.seeds:
        print(f"Running seed {seed}...")
        out_dir = run_one_seed(seed, args.profile, base_output_dir)
        scored = score_run(out_dir)
        if scored is None:
            print(f"  seed {seed}: no results produced, skipping")
            continue
        scored["seed"] = seed
        rows.append(scored)
        print(f"  seed {seed}: composite_score={scored['composite_score']}")

    if not rows:
        print("No successful runs.")
        return

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    summary_path = base_output_dir / "sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n=== TOP CANDIDATES ===")
    print(df[["seed", "composite_score", "monotonicity", "static_drop",
              "diversity_retention", "family_retention",
              "max_saturated_streak"]].head(5).to_string(index=False))
    print(f"\nFull results saved to {summary_path}")
    print(f"Best seed: {df.iloc[0]['seed']} "
          f"(composite_score={df.iloc[0]['composite_score']})")


if __name__ == "__main__":
    main()
