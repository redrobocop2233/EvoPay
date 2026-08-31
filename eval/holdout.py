"""Held-out evaluation for EVO-PAY closed loop.

Defines which attack families and primitive combinations are *excluded* from
the evolutionary loop and used only for final evaluation. This tests whether
Blue generalizes to genuinely unseen attack shapes, not just memorized ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random

from red_and_blue_team.red_team import (
    AttackGenome,
    GenomeCodec,
    CampaignPlanner,
    active_dimensions_of,
    attack_family_for,
    ACTIVE_THRESHOLD,
    GENE_NAMES,
    new_id,
)


@dataclass
class HoldoutConfig:
    """Config defining which attack regions are reserved for held-out eval."""

    # Families excluded from evolution entirely (matched by attack_family_for() output)
    held_out_families: tuple[str, ...] = ()

    # Primitive dimension tuples that must NOT co-occur in any genome during
    # the evolutionary loop. These combos are only instantiated for the final
    # held-out eval set.
    held_out_primitive_combos: tuple[tuple[str, ...], ...] = (
        ("geographic", "velocity"),
        ("device", "merchant", "coordination"),
    )


DEFAULT_HOLDOUT = HoldoutConfig()


def violates_holdout(strategy, holdout: HoldoutConfig) -> bool:
    """Check if an AttackStrategy matches any held-out family or combo."""
    active = set(active_dimensions_of(strategy))

    # Check family-level holdout
    family = attack_family_for(list(active), strategy)
    if family in holdout.held_out_families:
        return True

    # Check primitive-combination holdout
    for combo in holdout.held_out_primitive_combos:
        if set(combo).issubset(active):
            return True

    return False


def violates_holdout_genome(genome: AttackGenome, holdout: HoldoutConfig) -> bool:
    """Check if a genome would decode to a strategy violating holdout."""
    strategy = GenomeCodec.decode(genome)
    return violates_holdout(strategy, holdout)


def construct_genome_activating(combo: tuple[str, ...], rng: random.Random) -> AttackGenome:
    """Build a genome with specific primitives above ACTIVE_THRESHOLD.

    Genes in `combo` are set to high values (0.7-0.95), others are set low
    (0.1-0.5) to keep them below ACTIVE_THRESHOLD. This mirrors
    GenomeCodec.decode() logic in reverse.
    """
    gene_values = {}
    for gene in GENE_NAMES:
        if gene in combo:
            # Activate this dimension — set above threshold with some variation
            gene_values[gene] = ACTIVE_THRESHOLD + rng.uniform(0.05, 0.30)
        else:
            # Keep below threshold
            gene_values[gene] = rng.uniform(0.05, ACTIVE_THRESHOLD - 0.10)

    return AttackGenome(
        genome_id=new_id("holdout"),
        parent_id="holdout_eval",
        generation=999,
        **gene_values,
    )


def build_holdout_eval_set(
    planner: CampaignPlanner,
    ecosystem,
    holdout: HoldoutConfig,
    rng: random.Random,
    n_per_combo: int = 10,
) -> list[tuple[str, list]]:
    """Generate campaigns that ONLY use held-out families/combos.

    Returns list of (campaign_id, transactions) tuples.
    """
    campaigns = []

    for combo in holdout.held_out_primitive_combos:
        for _ in range(n_per_combo):
            genome = construct_genome_activating(combo, rng)
            strategy = GenomeCodec.decode(genome)
            customer = rng.choice(ecosystem.customers)
            campaign_id, txns = planner.build(customer, strategy)
            campaigns.append((campaign_id, txns))

    # Shuffle so they're not grouped by combo
    rng.shuffle(campaigns)
    return campaigns
