"""Metrics for the EVO-PAY adversarial closed loop.

These metrics deliberately distinguish three things that are often conflated:
- Blue detection performance (caught vs missed attacks)
- Red search progress (fitness/evasion/novelty)
- Synthetic quality (whether generated campaigns remain plausible and diverse)
"""
from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Iterable


def _distance(a, b):
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def generation_metrics(records, generation_stats=None):
    """Return metrics grouped by generation from StrategyMemory records."""
    groups = {}
    for r in records:
        groups.setdefault(r.generation, []).append(r)

    output = []
    for generation in sorted(groups):
        rows = groups[generation]
        vectors = [r.genome_vector for r in rows if r.genome_vector]
        signatures = {tuple(r.active_dimensions) for r in rows}
        families = {getattr(r, "attack_family", "unknown") for r in rows}
        risks = [r.risk_score for r in rows]
        fitness = [r.fitness for r in rows]

        pairwise = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pairwise.append(_distance(vectors[i], vectors[j]))

        # A transparent proxy for simulation fidelity. Every evaluated genome
        # passed the hard realism gate; this additional score rewards campaigns
        # that are not dominated by maximum-intensity genes.
        fidelity = []
        for r in rows:
            active = [v for v in (r.genome_vector or []) if v >= 0.65]
            intensity = _mean(active) if active else 0.0
            fidelity.append(max(0.0, 1.0 - max(0.0, intensity - 0.65) / 0.35))

        caught = sum(bool(r.detected) for r in rows)
        metric = {
            "generation": generation,
            "campaigns": len(rows),
            "detection_rate": round(caught / len(rows), 4) if rows else 0.0,
            "attack_success_rate": round(1 - caught / len(rows), 4) if rows else 0.0,
            "mean_risk": round(_mean(risks), 4),
            "mean_fitness": round(_mean(fitness), 4),
            "mean_novelty": round(_mean(r.novelty_score for r in rows), 4),
            "unique_attack_families": len(families),
            "unique_behavior_signatures": len(signatures),
            "genome_diversity": round(_mean(pairwise), 4),
            "fidelity_proxy": round(_mean(fidelity), 4),
            "hard_realism_pass_rate": 1.0,
        }
        output.append(metric)

    return output


def overall_metrics(records):
    """Compact aggregate metrics suitable for the competition write-up."""
    if not records:
        return {}
    caught = sum(bool(r.detected) for r in records)
    families = {getattr(r, "attack_family", "unknown") for r in records}
    signatures = {tuple(r.active_dimensions) for r in records}
    return {
        "campaigns_evaluated": len(records),
        "detection_rate": round(caught / len(records), 4),
        "attack_success_rate": round(1 - caught / len(records), 4),
        "mean_risk": round(_mean(r.risk_score for r in records), 4),
        "mean_fitness": round(_mean(r.fitness for r in records), 4),
        "unique_attack_families": len(families),
        "unique_behavior_signatures": len(signatures),
        "generations": len({r.generation for r in records}),
    }
