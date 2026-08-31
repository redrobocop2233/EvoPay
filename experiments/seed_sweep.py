"""Seed sweep experiment runner for EVO-PAY.

Runs the closed loop across multiple seeds at demo scale, scores each run for
how well it supports the "static degrades, adaptive holds" narrative and how
well diversity is preserved, and ranks them to select the cleanest seed for
reproducible submission and demo artifacts.

Usage:
    python -m experiments.seed_sweep --seeds 1 7 13 21 33 42 55 71 88 99 --profile demo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run_one_seed(seed: int, profile: str, base_output_dir: Path) -> Path:
    """Execute one closed-loop run for a given seed."""
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
    """Score a completed run on narrative clarity, diversity, and saturation metrics."""
    static_path = out_dir / "static_vs_adaptive.json"
    gen_stats_path = out_dir / "generation_stats.csv"
    summary_path = out_dir / "closed_loop_summary.json"

    if not static_path.exists() or not gen_stats_path.exists():
        return None

    static_data = json.loads(static_path.read_text(encoding="utf-8"))
    if not static_data:
        return None

    static_rates = [row["static_detection_rate"] for row in static_data]
    adaptive_rates = [row["adaptive_detection_rate"] for row in static_data]

    # Monotonicity score: fraction of consecutive steps where static declines (or stays flat)
    declines = sum(
        1 for a, b in zip(static_rates, static_rates[1:]) if b <= a
    )
    monotonicity = declines / max(1, len(static_rates) - 1)

    # Total drop from first to last generation (bigger = more dramatic degradation of static defense)
    static_drop = static_rates[0] - static_rates[-1]

    # Adaptive should stay high/stable — penalize big swings
    adaptive_stability = 1.0 - float(np.std(adaptive_rates)) if adaptive_rates else 0.0

    # Diversity metrics
    gen_stats = pd.read_csv(gen_stats_path)
    # Check for diversity columns
    diversity_col = "genome_diversity" if "genome_diversity" in gen_stats.columns else ("diversity" if "diversity" in gen_stats.columns else None)
    if diversity_col:
        diversity_start = float(gen_stats[diversity_col].iloc[0])
        diversity_end = float(gen_stats[diversity_col].iloc[-1])
        diversity_retention = diversity_end / max(diversity_start, 1e-6)
    else:
        diversity_retention = 1.0

    family_col = "unique_attack_families" if "unique_attack_families" in gen_stats.columns else ("families" if "families" in gen_stats.columns else None)
    if family_col:
        families_start = float(gen_stats[family_col].iloc[0])
        families_end = float(gen_stats[family_col].iloc[-1])
        family_retention = families_end / max(families_start, 1.0)
    else:
        family_retention = 1.0

    # Saturation streak: count consecutive generations with ~100% adaptive detection
    saturated_streak = 0
    max_saturated_streak = 0
    for rate in adaptive_rates:
        if rate >= 0.999:
            saturated_streak += 1
            max_saturated_streak = max(max_saturated_streak, saturated_streak)
        else:
            saturated_streak = 0

    # Composite score
    composite = (
        0.30 * monotonicity +
        0.20 * min(max(static_drop, 0.0), 1.0) +
        0.20 * min(diversity_retention, 1.5) +
        0.15 * min(family_retention, 1.5) +
        0.15 * max(0.0, 1.0 - 0.25 * max_saturated_streak)
    )

    # Optional held-out evaluation summary
    holdout_det_rate = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            holdout_det_rate = summary.get("holdout_metrics", {}).get("detection_rate")
        except Exception:
            pass

    return {
        "monotonicity": round(monotonicity, 3),
        "static_drop": round(static_drop, 3),
        "adaptive_stability": round(adaptive_stability, 3),
        "diversity_retention": round(diversity_retention, 3),
        "family_retention": round(family_retention, 3),
        "max_saturated_streak": max_saturated_streak,
        "holdout_detection_rate": holdout_det_rate,
        "composite_score": round(composite, 4),
        "static_rates": static_rates,
        "adaptive_rates": adaptive_rates,
    }


def main():
    parser = argparse.ArgumentParser(description="EVO-PAY Seed Sweep across random seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 13, 21, 33, 42, 55, 71, 88, 99])
    parser.add_argument("--profile", default="demo", choices=["demo", "quick"])
    parser.add_argument("--output-dir", default="experiments/seed_sweep")
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"=== Starting EVO-PAY Seed Sweep ({len(args.seeds)} seeds, profile={args.profile}) ===")
    for seed in args.seeds:
        print(f"Running seed {seed}...")
        out_dir = run_one_seed(seed, args.profile, base_output_dir)
        scored = score_run(out_dir)
        if scored is None:
            print(f"  seed {seed}: no results produced, skipping")
            continue
        scored["seed"] = seed
        rows.append(scored)
        print(f"  seed {seed}: composite_score={scored['composite_score']:.4f} "
              f"(static_drop={scored['static_drop']:.2f}, monotonicity={scored['monotonicity']:.2f}, "
              f"max_sat_streak={scored['max_saturated_streak']})")

    if not rows:
        print("No successful runs.")
        return

    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    summary_path = base_output_dir / "sweep_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n=== TOP CANDIDATE SEEDS ===")
    display_cols = [
        "seed", "composite_score", "monotonicity", "static_drop",
        "diversity_retention", "family_retention", "max_saturated_streak"
    ]
    if "holdout_detection_rate" in df.columns:
        display_cols.append("holdout_detection_rate")
    print(df[display_cols].head(5).to_string(index=False))
    print(f"\nFull results saved to: {summary_path}")
    best_seed = int(df.iloc[0]["seed"])
    best_score = float(df.iloc[0]["composite_score"])
    print(f"\n🏆 Best seed: {best_seed} (composite_score={best_score:.4f})")


if __name__ == "__main__":
    main()
