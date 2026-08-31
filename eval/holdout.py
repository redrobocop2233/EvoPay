"""Held-out attack evaluation module for EVO-PAY.

Splits the attack-strategy space into a training region (available to Red Team
evolution) and a held-out evaluation region (reserved primitive combinations
and families tested only at final evaluation).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from red_and_blue_team.ecosystem import PaymentEcosystem, new_id
from red_and_blue_team.red_team import (
    AttackGenome,
    AttackStrategy,
    GenomeCodec,
    CampaignPlanner,
    RealismValidator,
    GENE_NAMES,
    ACTIVE_THRESHOLD,
    active_dimensions_of,
    attack_family_for,
)


@dataclass
class HoldoutConfig:
    """Configuration for reserving attack strategies from the evolutionary loop."""

    # Families excluded from the evolutionary loop entirely; reserved for final holdout eval.
    held_out_families: tuple[str, ...] = ()

    # Primitive tuples that must NOT co-occur in any genome the evolutionary loop is allowed
    # to select/mutate toward. These combos are only ever instantiated for the final held-out eval set.
    held_out_primitive_combos: tuple[tuple[str, ...], ...] = (
        ("geographic", "velocity"),
        ("device", "merchant", "coordination"),
    )


DEFAULT_HOLDOUT = HoldoutConfig()


def violates_holdout(
    candidate: AttackGenome | AttackStrategy,
    holdout: HoldoutConfig = DEFAULT_HOLDOUT,
) -> bool:
    """Check if a genome or strategy violates the holdout constraints.

    Returns True if the candidate matches a held-out family or contains a held-out
    primitive combination, meaning it must be excluded from evolutionary training.
    """
    if holdout is None:
        return False

    if isinstance(candidate, AttackGenome):
        strategy = GenomeCodec.decode(candidate)
    else:
        strategy = candidate

    active = set(active_dimensions_of(strategy))
    family = attack_family_for(list(active), strategy)

    if family in holdout.held_out_families:
        return True

    for combo in holdout.held_out_primitive_combos:
        if set(combo).issubset(active):
            return True

    return False


def construct_genome_activating(
    combo: tuple[str, ...] | list[str],
    rng: random.Random,
    generation: int = 999,
) -> AttackGenome:
    """Construct an AttackGenome that specifically activates the given primitive combination.

    Genes in combo are assigned values above ACTIVE_THRESHOLD (0.65), while inactive
    genes are assigned values well below ACTIVE_THRESHOLD to ensure precise activation.
    """
    combo_set = set(combo)
    gene_values = {}

    for gene in GENE_NAMES:
        if gene in combo_set:
            # Active range [0.70, 0.90]
            gene_values[gene] = round(rng.uniform(ACTIVE_THRESHOLD + 0.05, 0.90), 4)
        else:
            # Inactive range [0.05, 0.45]
            gene_values[gene] = round(rng.uniform(0.05, ACTIVE_THRESHOLD - 0.20), 4)

    return AttackGenome(
        genome_id=new_id("hgen"),
        parent_id="holdout_eval",
        generation=generation,
        **gene_values,
    )


@dataclass
class HoldoutCampaign:
    """Container for a held-out evaluation campaign."""
    campaign_id: str
    transactions: list
    strategy: AttackStrategy
    genome: AttackGenome
    combo: tuple[str, ...]
    family: str


def build_holdout_eval_set(
    ecosystem: PaymentEcosystem,
    holdout: HoldoutConfig = DEFAULT_HOLDOUT,
    n_campaigns: int = 40,
    rng: random.Random | None = None,
    validator: RealismValidator | None = None,
) -> list[HoldoutCampaign]:
    """Generate a dedicated set of evaluation campaigns drawn strictly from the held-out space.

    These campaigns bypass evolutionary selection and test generalization to unseen
    attack primitive combinations.
    """
    if rng is None:
        rng = random.Random(42)
    if validator is None:
        validator = RealismValidator()

    planner = CampaignPlanner(ecosystem, rng)
    combos = holdout.held_out_primitive_combos
    if not combos:
        return []

    campaigns_per_combo = max(1, n_campaigns // len(combos))
    campaigns: list[HoldoutCampaign] = []

    for combo in combos:
        generated_for_combo = 0
        attempts = 0
        max_attempts = campaigns_per_combo * 20

        while generated_for_combo < campaigns_per_combo and attempts < max_attempts:
            attempts += 1
            genome = construct_genome_activating(combo, rng)
            strategy = GenomeCodec.decode(genome)

            if not validator.is_realistic(strategy):
                continue

            customer = rng.choice(ecosystem.customers)
            campaign_id, transactions = planner.build(customer, strategy)
            active = active_dimensions_of(strategy)
            family = attack_family_for(active, strategy)

            campaigns.append(
                HoldoutCampaign(
                    campaign_id=campaign_id,
                    transactions=transactions,
                    strategy=strategy,
                    genome=genome,
                    combo=combo,
                    family=family,
                )
            )
            generated_for_combo += 1

    return campaigns
