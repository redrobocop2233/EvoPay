from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

from red_and_blue_team.red_team import AttackGenome, GenomeCodec, active_dimensions_of, AttackStrategy

PRIMITIVE_TO_GENE = {
    "amount_spike": "amount",
    "card_testing": "amount",
    "odd_hour": "temporal",
    "device_switch": "device",
    "geo_anomaly": "geographic",
    "category_drift": "merchant",
    "velocity_burst": "velocity",
    "multi_account_coordination": "coordination",
}

@dataclass(frozen=True)
class HoldoutConfig:
    held_out_families: tuple[str, ...] = ()
    held_out_primitive_combos: tuple[tuple[str, ...], ...] = (
        ("geo_anomaly", "velocity_burst"),
        ("device_switch", "category_drift", "multi_account_coordination"),
    )
    active_threshold: float = 0.70

DEFAULT_HOLDOUT = HoldoutConfig()

def violates_holdout(strategy: AttackStrategy, holdout: HoldoutConfig = DEFAULT_HOLDOUT) -> bool:
    family = None
    # Family labels are assigned by Red Team from active dimensions. Import lazily
    # to avoid making eval a second source of family definitions.
    from red_and_blue_team.red_team import attack_family_for
    active = active_dimensions_of(strategy)
    family = attack_family_for(active, strategy)
    if family in holdout.held_out_families:
        return True
    primitive_map = {
        "amount": "amount_spike", "temporal": "odd_hour", "device": "device_switch",
        "geographic": "geo_anomaly", "merchant": "category_drift",
        "velocity": "velocity_burst", "coordination": "multi_account_coordination",
    }
    active_primitives = {primitive_map[d] for d in active}
    return any(set(combo).issubset(active_primitives) for combo in holdout.held_out_primitive_combos)

def construct_genome_activating(combo: tuple[str, ...], rng: random.Random, generation: int = 999) -> AttackGenome:
    values = {d: rng.uniform(0.02, 0.35) for d in PRIMITIVE_TO_GENE.values()}
    for primitive in combo:
        gene = PRIMITIVE_TO_GENE[primitive]
        values[gene] = rng.uniform(0.76, 0.95)
    return AttackGenome(
        genome_id=f"holdout_{rng.randrange(10**12):012d}",
        generation=generation,
        **values,
    )

def build_eval_set(controller, holdout: HoldoutConfig, n_campaigns: int, rng: random.Random):
    campaigns = []
    combos = list(holdout.held_out_primitive_combos)
    if not combos:
        return campaigns
    per_combo = max(1, n_campaigns // len(combos))
    for combo in combos:
        for _ in range(per_combo):
            genome = construct_genome_activating(combo, rng)
            strategy = GenomeCodec.decode(genome)
            if violates_holdout(strategy, holdout):
                customer = rng.choice(controller.eco.customers)
                campaign_id, txns = controller.planner.build(customer, strategy)
                campaigns.append((genome, strategy, campaign_id, txns))
    return campaigns
