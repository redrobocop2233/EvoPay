"""EVO-PAY closed-loop adversarial evaluation.

The loop is deliberately separated into four stages:

    1. Gemini discovers hypotheses from the threat landscape + Blue blind spots.
    2. Red compiles them to genomes and evolves against the actual detector.
    3. Blue decisions are measured, including false-positive checks on untouched
       legitimate traffic.
    4. Gemini autopsies failures and proposes bounded mutations; optional Blue
       retraining consumes the accumulated adversarial feed between rounds.

GenAI never writes transaction rows and never changes a Blue risk score.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time as _time
from pathlib import Path

import numpy as np

from red_and_blue_team.ecosystem import PaymentEcosystem
from red_and_blue_team.blue_team import HeuristicDetector, TrainableDetector
from red_and_blue_team.blue_team_client import BlueTeamClient, wait_for_api
from red_and_blue_team.red_team import (
    AttackGenome,
    GenomeCodec,
    FailureAnalyzer,
    RedTeamController,
    active_dimensions_of,
    random_genome,
)
from eval.metrics import compute_classification_metrics, recall_at_fpr
from eval.closed_loop_metrics import generation_metrics, overall_metrics
from eval.holdout import HoldoutConfig, DEFAULT_HOLDOUT, build_holdout_eval_set


GENES = {"amount", "temporal", "device", "geographic", "merchant", "velocity", "coordination"}

DETECTION_THRESHOLD = 0.5

# Demo profile presets (Section 2)
PROFILES = {
    "demo": {"generations": 8, "population": 40, "discover": 6},
    "quick": {"generations": 3, "population": 12, "discover": 4},
}


class GuidedMutationEngine:
    """Translate GenAI qualitative advice into bounded local genome edits."""

    STEP = 0.10

    def apply(self, genome: AttackGenome, mutations: list[dict], generation: int) -> AttackGenome:
        child = AttackGenome(
            genome_id=f"genai_{genome.genome_id}_{generation}",
            amount=genome.amount,
            temporal=genome.temporal,
            device=genome.device,
            geographic=genome.geographic,
            merchant=genome.merchant,
            velocity=genome.velocity,
            coordination=genome.coordination,
            parent_id=genome.genome_id,
            generation=generation,
        )
        for mutation in mutations[:3]:
            dim = mutation.get("dimension")
            direction = mutation.get("direction")
            if dim not in GENES:
                continue
            value = getattr(child, dim)
            if direction in ("increase", "add"):
                value += self.STEP
            elif direction in ("decrease", "remove"):
                value -= self.STEP
            setattr(child, dim, max(0.0, min(1.0, value)))
        return child


def _strategy_summary(genome: AttackGenome) -> dict:
    strategy = GenomeCodec.decode(genome)
    return {
        "generation": genome.generation,
        "active_dimensions": active_dimensions_of(strategy),
        "temporal_pattern": strategy.temporal_pattern,
        "amount_pattern": strategy.amount_pattern,
        "device_pattern": strategy.device_pattern,
        "geographic_pattern": strategy.geographic_pattern,
        "merchant_pattern": strategy.merchant_pattern,
        "velocity_pattern": strategy.velocity_pattern,
        "coordination_pattern": strategy.coordination_pattern,
    }


def _record_genome(record):
    return AttackGenome(
        genome_id=record.genome_id,
        amount=record.genome_vector[0],
        temporal=record.genome_vector[1],
        device=record.genome_vector[2],
        geographic=record.genome_vector[3],
        merchant=record.genome_vector[4],
        velocity=record.genome_vector[5],
        coordination=record.genome_vector[6],
        parent_id=record.parent_id,
        generation=record.generation,
    )


def _safe_identify_excerpt():
    path = Path(__file__).resolve().parent.parent / "IDENTIFY.md"
    if not path.exists():
        return "No IDENTIFY.md available."
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:12000]


def _build_genai():
    from genai.client import GenAIClient
    from genai.discoverer import ThreatDiscoverer
    from genai.analyst import AttackAnalyst
    client = GenAIClient()
    return ThreatDiscoverer(client), AttackAnalyst(client)


def _evaluate_legitimate(detector, eco, rng, n=25):
    if not eco.transactions:
        return []
    sample = rng.sample(eco.transactions, min(n, len(eco.transactions)))
    scores = []
    for txn in sample:
        try:
            result = detector.evaluate([txn], eco)
            scores.append(float(result.risk_score))
        except Exception:
            continue
    return scores


def _holdout_attacks(controller, detector, eco, rng, n=20):
    """Fresh genomes evaluated after the loop; they receive no GenAI feedback."""
    scores = []
    detected = 0
    for _ in range(n):
        genome = random_genome(rng, generation=999)
        strategy = GenomeCodec.decode(genome)
        if not controller.validator.is_realistic(strategy):
            continue
        customer = rng.choice(eco.customers)
        _, txns = controller.planner.build(customer, strategy)
        result = detector.evaluate(txns, eco)
        scores.append(float(result.risk_score))
        detected += int(result.detected)
    return {
        "n": len(scores),
        "detection_rate": round(detected / len(scores), 4) if scores else None,
        "mean_risk": round(float(np.mean(scores)), 4) if scores else None,
    }


def run_loop(
    generations=3,
    population=12,
    discover=4,
    customers=120,
    merchants=40,
    days=60,
    api_url="http://127.0.0.1:8000",
    seed=7,
    use_genai=True,
    retrain_blue_every=0,
    output_dir="integration/results",
    legit_samples=25,
    holdout_attacks=20,
    use_trainable=True,
    use_legacy_proxy=False,
):
    rng = random.Random(seed)
    eco = PaymentEcosystem(customers, merchants, days)
    eco.generate_transactions()

    using_api = bool(api_url)
    if using_api:
        wait_for_api(api_url)
        detector = BlueTeamClient(
            base_url=api_url,
            timeout=15,
            fallback=HeuristicDetector(),
        )
    elif use_legacy_proxy:
        detector = HeuristicDetector()
    else:
        # Section 1: Use the TrainableDetector for detector-aware fitness
        detector = TrainableDetector(threshold=DETECTION_THRESHOLD)
        # Seed the trainable detector with legitimate transaction baselines
        detector.add_legitimate_baseline(eco, rng, n=80)
        use_trainable = True

    # Section 6: Create a frozen static detector snapshot for two-curve comparison
    static_detector = None
    if use_trainable and not using_api:
        # The static detector is a copy of the initial state — never retrained
        static_detector = copy.deepcopy(detector)

    discoverer = analyst = None
    if use_genai:
        discoverer, analyst = _build_genai()

    controller = RedTeamController(
        eco,
        detector=detector,
        seed=seed,
        population_size=population,
        holdout=DEFAULT_HOLDOUT,
    )
    guided = GuidedMutationEngine()
    discoveries = []
    autopsies = []
    retrains = []
    loop_events = []
    seed_genomes = []
    static_vs_adaptive = []   # Section 6: per-generation two-curve data
    latency_samples = []      # Section 4: latency instrumentation
    identify_excerpt = _safe_identify_excerpt()

    if discoverer:
        hypotheses, research_only = discoverer.discover(
            n=discover,
            known_families=[],
            weak_dimensions=None,
        )
        discoveries.extend(h.model_dump() for h in hypotheses + research_only)
        from genai.discoverer import hypothesis_to_genome
        seed_genomes.extend(hypothesis_to_genome(h, rng) for h in hypotheses)
        loop_events.append({
            "event": "initial_discovery",
            "simulatable": len(hypotheses),
            "research_only": len(research_only),
        })

    for round_index in range(generations):
        # Section 4: measure evaluation latency
        t_gen_start = _time.perf_counter()

        controller.evolve(generations=1, seed_genomes=seed_genomes)
        generation = controller.next_generation_number - 1
        current = [r for r in controller.memory.records if r.generation == generation]
        current.sort(key=lambda r: (r.detected, r.risk_score))

        t_gen_end = _time.perf_counter()
        gen_latency_ms = (t_gen_end - t_gen_start) * 1000
        if current:
            latency_samples.append(gen_latency_ms / len(current))  # per-campaign latency

        loop_events.append({
            "event": "blue_evaluation",
            "generation": generation,
            "campaigns": len(current),
            "detected": sum(int(r.detected) for r in current),
            "latency_ms": round(gen_latency_ms, 1),
        })

        # Section 6: Score this generation's campaigns against the static detector
        if static_detector and current:
            static_detected = 0
            adaptive_detected = 0
            for record in current:
                # Adaptive detection was already computed during evolve()
                if record.detected:
                    adaptive_detected += 1
                # Score against static (frozen) detector
                txns = controller.fraud_transactions_by_campaign.get(record.campaign_id)
                if txns:
                    static_result = static_detector.evaluate(txns, eco)
                    if static_result.detected:
                        static_detected += 1

            static_vs_adaptive.append({
                "generation": generation,
                "static_detection_rate": round(static_detected / len(current), 4),
                "adaptive_detection_rate": round(adaptive_detected / len(current), 4),
            })

        seed_genomes = []

        # Autopsy prioritizes attacks Blue missed, then low-risk caught attacks.
        if analyst and current:
            selected = current[:3]
            for record in selected:
                genome = _record_genome(record)
                autopsy = analyst.analyze(
                    record.genome_id,
                    _strategy_summary(genome),
                    {
                        "risk_score": record.risk_score,
                        "detected": record.detected,
                        "reason_codes": record.reason_codes,
                    },
                )
                autopsies.append(autopsy.model_dump())
                if autopsy.recommended_mutations:
                    seed_genomes.append(
                        guided.apply(
                            genome,
                            [m.model_dump() for m in autopsy.recommended_mutations],
                            generation + 1,
                        )
                    )

            weakness_report = FailureAnalyzer().analyze(controller.memory)
            weak_dimensions = [
                dim for dim, stats in weakness_report.items()
                if stats.get("active", 0) >= 2 and stats.get("detection_rate") is not None
            ]
            weak_dimensions.sort(key=lambda d: weakness_report[d]["detection_rate"])

            hypotheses, research_only = discoverer.discover(
                n=2,
                known_families=sorted({getattr(r, "attack_family", "unknown") for r in current}),
                weak_dimensions=weak_dimensions[:3],
            )
            discoveries.extend(h.model_dump() for h in hypotheses + research_only)
            from genai.discoverer import hypothesis_to_genome
            seed_genomes.extend(hypothesis_to_genome(h, rng) for h in hypotheses)
            loop_events.append({
                "event": "adaptive_discovery",
                "generation": generation,
                "weak_dimensions": weak_dimensions[:3],
                "simulatable": len(hypotheses),
                "research_only": len(research_only),
            })

        # Section 1: Retrain the trainable detector after each generation
        if use_trainable and hasattr(detector, 'update') and not using_api:
            detector.update(current, controller.fraud_transactions_by_campaign, eco)
            loop_events.append({
                "event": "blue_retrain_trainable",
                "generation": generation,
                "training_samples": len(detector._training_X),
            })

        # Optional explicit Blue retraining for API-based detector (original mechanism)
        elif retrain_blue_every and (generation + 1) % retrain_blue_every == 0:
            if using_api:
                from integration.adversarial_retrain import retrain_blue, reload_api
                report = retrain_blue(
                    Path(__file__).resolve().parent.parent / "data" / "raw" / "synthetic_transactions.csv",
                    controller.fraud_transactions,
                    Path(output_dir),
                )
                report["api_reloaded"] = reload_api(api_url)
                retrains.append(report)
                loop_events.append({"event": "blue_retrain", "generation": generation, **report})
            else:
                loop_events.append({
                    "event": "blue_retrain_skipped",
                    "generation": generation,
                    "reason": "local heuristic detector has no trainable state",
                })

    # Untouched legitimate traffic is a false-positive check.
    legit_scores = _evaluate_legitimate(detector, eco, rng, legit_samples)

    # Section A — Held-out evaluation on attack families/combos never seen during evolution
    holdout_campaigns = build_holdout_eval_set(
        ecosystem=eco,
        holdout=DEFAULT_HOLDOUT,
        n_campaigns=holdout_attacks,
        rng=rng,
        validator=controller.validator,
    )
    holdout_results = []
    for camp in holdout_campaigns:
        res = detector.evaluate(camp.transactions, eco)
        holdout_results.append({
            "campaign_id": camp.campaign_id,
            "combo": list(camp.combo),
            "family": camp.family,
            "risk_score": float(res.risk_score),
            "detected": bool(res.detected),
        })

    holdout_detected = sum(1 for r in holdout_results if r["detected"])
    holdout_risk_scores = [r["risk_score"] for r in holdout_results]

    y_holdout_true = np.array([1] * len(holdout_risk_scores) + [0] * len(legit_scores))
    y_holdout_scores = np.array(holdout_risk_scores + legit_scores)
    y_holdout_pred = (y_holdout_scores >= 0.5).astype(int) if len(y_holdout_scores) else np.array([])
    holdout_classification = (
        compute_classification_metrics(y_holdout_true, y_holdout_pred, y_holdout_scores)
        if len(y_holdout_scores) and len(set(y_holdout_true)) == 2
        else {}
    )
    if holdout_classification:
        holdout_classification["recall_at_1pct_fpr"] = recall_at_fpr(y_holdout_true, y_holdout_scores, 0.01)
        holdout_classification["recall_at_5pct_fpr"] = recall_at_fpr(y_holdout_true, y_holdout_scores, 0.05)

    holdout_metrics = {
        "description": "Performance on attack families/primitive-combos never seen during evolution",
        "n_campaigns": len(holdout_results),
        "detection_rate": round(holdout_detected / len(holdout_results), 4) if holdout_results else None,
        "attack_success_rate": round(1.0 - (holdout_detected / len(holdout_results)), 4) if holdout_results else None,
        "mean_risk": round(float(np.mean(holdout_risk_scores)), 4) if holdout_risk_scores else None,
        "held_out_primitive_combos": [list(c) for c in DEFAULT_HOLDOUT.held_out_primitive_combos],
        "held_out_families": list(DEFAULT_HOLDOUT.held_out_families),
        **holdout_classification,
    }

    records = controller.memory.records
    y_true = np.array([1] * len(records) + [0] * len(legit_scores))
    y_scores = np.array([r.risk_score for r in records] + legit_scores)
    y_pred = (y_scores >= 0.5).astype(int) if len(y_scores) else np.array([])
    classification = compute_classification_metrics(y_true, y_pred, y_scores) if len(y_scores) and len(set(y_true)) == 2 else {}
    if classification:
        classification["recall_at_1pct_fpr"] = recall_at_fpr(y_true, y_scores, 0.01)
        classification["recall_at_5pct_fpr"] = recall_at_fpr(y_true, y_scores, 0.05)
    classification["n_fraud_campaigns"] = len(records)
    classification["n_legitimate_samples"] = len(legit_scores)
    classification["mean_legit_risk"] = round(float(np.mean(legit_scores)), 4) if legit_scores else None

    # Section 4: latency statistics
    latency_stats = {}
    if latency_samples:
        latency_stats = {
            "p50_ms": round(float(np.percentile(latency_samples, 50)), 2),
            "p95_ms": round(float(np.percentile(latency_samples, 95)), 2),
            "mean_ms": round(float(np.mean(latency_samples)), 2),
            "samples": len(latency_samples),
        }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    controller.to_csv(out)

    timeline = generation_metrics(records)

    # Merge static_vs_adaptive data into the timeline
    sva_by_gen = {item["generation"]: item for item in static_vs_adaptive}
    for entry in timeline:
        gen = entry["generation"]
        if gen in sva_by_gen:
            entry["static_detection_rate"] = sva_by_gen[gen]["static_detection_rate"]
            entry["adaptive_detection_rate"] = sva_by_gen[gen]["adaptive_detection_rate"]

    summary = {
        "overall_red_blue": overall_metrics(records),
        "blue_classification": classification,
        "holdout_metrics": holdout_metrics,
        "holdout_attack_eval": holdout_metrics,
        "generation_timeline": timeline,
        "static_vs_adaptive": static_vs_adaptive,
        "latency": latency_stats,
        "discoveries": len(discoveries),
        "autopsies": len(autopsies),
        "retrains": len(retrains),
        "genai_enabled": use_genai,
        "blue_api": api_url or None,
        "detector_type": type(detector).__name__,
        "seed": seed,
    }

    (out / "genai_discoveries.json").write_text(json.dumps(discoveries, indent=2), encoding="utf-8")
    (out / "genai_autopsies.json").write_text(json.dumps(autopsies, indent=2), encoding="utf-8")
    (out / "retraining_history.json").write_text(json.dumps(retrains, indent=2), encoding="utf-8")
    (out / "loop_events.json").write_text(json.dumps(loop_events, indent=2), encoding="utf-8")
    (out / "closed_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if static_vs_adaptive:
        (out / "static_vs_adaptive.json").write_text(json.dumps(static_vs_adaptive, indent=2), encoding="utf-8")
    return controller, discoveries, autopsies, summary


def main():
    parser = argparse.ArgumentParser(description="Run EVO-PAY's adaptive GenAI Red/Blue loop")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--discover", type=int, default=4)
    parser.add_argument("--customers", type=int, default=120)
    parser.add_argument("--merchants", type=int, default=40)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--local-blue", action="store_true", help="Use the local trainable detector instead of Blue API")
    parser.add_argument("--no-genai", action="store_true")
    parser.add_argument("--use-legacy-proxy", action="store_true", help="Use HeuristicDetector instead of TrainableDetector (no retraining)")
    parser.add_argument("--retrain-blue-every", type=int, default=0, help="Retrain/reload Blue every N generations; 0 disables")
    parser.add_argument("--legit-samples", type=int, default=25)
    parser.add_argument("--holdout-attacks", type=int, default=20)
    parser.add_argument("--output-dir", default="integration/results")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default=None,
                        help="Use a preset profile (overrides generations/population/discover)")
    args = parser.parse_args()

    # Section 2: Apply profile presets
    if args.profile:
        preset = PROFILES[args.profile]
        args.generations = preset["generations"]
        args.population = preset["population"]
        args.discover = preset["discover"]

    api_url = "" if args.local_blue else args.api_url
    controller, discoveries, autopsies, summary = run_loop(
        generations=args.generations,
        population=args.population,
        discover=args.discover,
        customers=args.customers,
        merchants=args.merchants,
        days=args.days,
        api_url=api_url,
        seed=args.seed,
        use_genai=not args.no_genai,
        retrain_blue_every=args.retrain_blue_every,
        output_dir=args.output_dir,
        legit_samples=args.legit_samples,
        holdout_attacks=args.holdout_attacks,
        use_legacy_proxy=args.use_legacy_proxy,
    )

    print("\nEVO-PAY ADAPTIVE CLOSED LOOP")
    print("generation | detection | attack_success | avg_fitness | avg_risk | families | diversity")
    for stat in summary["generation_timeline"]:
        print(
            f"{stat['generation']:>10} | {stat['detection_rate']:>9.1%} | "
            f"{stat['attack_success_rate']:>14.1%} | {stat['mean_fitness']:>11.4f} | "
            f"{stat['mean_risk']:>8.4f} | {stat['unique_attack_families']:>8} | "
            f"{stat['genome_diversity']:>8.4f}"
        )

    # Section 6: Print static vs adaptive comparison
    if summary.get("static_vs_adaptive"):
        print("\nSTATIC vs. ADAPTIVE DETECTION (two-curve comparison)")
        print("generation | static_det | adaptive_det")
        for entry in summary["static_vs_adaptive"]:
            print(
                f"{entry['generation']:>10} | {entry['static_detection_rate']:>10.1%} | "
                f"{entry['adaptive_detection_rate']:>12.1%}"
            )

    bm = summary["blue_classification"]
    print("\nBLUE IN-DISTRIBUTION PERFORMANCE (Evolved Red + Untouched Legit)")
    for key in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"):
        if key in bm:
            print(f"  {key:22}: {bm[key]:.4f}")

    hm = summary.get("holdout_metrics", {})
    if hm:
        print("\nBLUE HELD-OUT GENERALIZATION (Unseen Primitive Combinations + Legit)")
        print(f"  held-out campaigns    : {hm.get('n_campaigns', 0)}")
        if hm.get("detection_rate") is not None:
            print(f"  detection_rate        : {hm['detection_rate']:.4f}")
            print(f"  attack_success_rate   : {hm['attack_success_rate']:.4f}")
            print(f"  mean_risk             : {hm['mean_risk']:.4f}")
        for key in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"):
            if key in hm:
                print(f"  {key:22}: {hm[key]:.4f}")

    # Section 4: Print latency stats
    if summary.get("latency"):
        lat = summary["latency"]
        print(f"\nLATENCY (per campaign, {lat['samples']} samples)")
        print(f"  p50: {lat['p50_ms']:.1f}ms  p95: {lat['p95_ms']:.1f}ms  mean: {lat['mean_ms']:.1f}ms")

    print(f"\nDetector: {summary.get('detector_type', 'unknown')}")
    print(f"GenAI discoveries: {len(discoveries)}")
    print(f"GenAI autopsies:   {len(autopsies)}")
    print(f"Results: {args.output_dir}/")

    # Section 2: Save experiment config for reproducibility
    if args.profile:
        experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "profile": args.profile,
            "seed": args.seed,
            "generations": args.generations,
            "population": args.population,
            "discover": args.discover,
            "customers": args.customers,
            "merchants": args.merchants,
            "days": args.days,
            "use_genai": not args.no_genai,
            "use_legacy_proxy": args.use_legacy_proxy,
            "detector_type": summary.get("detector_type"),
        }
        config_path = experiments_dir / f"demo_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        # Save the full summary as the frozen run artifact
        run_path = experiments_dir / f"demo_run_final.json"
        run_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nExperiment saved: {config_path} + {run_path}")


if __name__ == "__main__":
    main()
