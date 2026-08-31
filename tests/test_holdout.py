"""Unit tests for held-out evaluation and seed sweep components."""
import random
import unittest

from red_and_blue_team.ecosystem import PaymentEcosystem
from red_and_blue_team.red_team import (
    AttackGenome,
    GenomeCodec,
    MutationEngine,
    EvolutionEngine,
    RedTeamController,
    active_dimensions_of,
)
from eval.holdout import (
    HoldoutConfig,
    DEFAULT_HOLDOUT,
    violates_holdout,
    construct_genome_activating,
    build_holdout_eval_set,
)


class TestHoldoutEvaluation(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(42)
        self.holdout = HoldoutConfig(
            held_out_families=("custom_test_family",),
            held_out_primitive_combos=(
                ("geographic", "velocity"),
                ("device", "merchant", "coordination"),
            ),
        )

    def test_construct_genome_activating(self):
        combo = ("geographic", "velocity")
        genome = construct_genome_activating(combo, self.rng)
        strategy = GenomeCodec.decode(genome)
        active = active_dimensions_of(strategy)

        # Combo dimensions must be active
        self.assertIn("geographic", active)
        self.assertIn("velocity", active)
        # Non-combo dimensions must be inactive
        self.assertNotIn("device", active)
        self.assertNotIn("coordination", active)

    def test_violates_holdout_detection(self):
        # Genome activating geographic + velocity
        violating_genome = construct_genome_activating(("geographic", "velocity"), self.rng)
        self.assertTrue(violates_holdout(violating_genome, self.holdout))

        # Genome activating only amount
        safe_genome = construct_genome_activating(("amount",), self.rng)
        self.assertFalse(violates_holdout(safe_genome, self.holdout))

    def test_evolution_engine_respects_holdout(self):
        engine = EvolutionEngine(self.rng, population_size=15, holdout=self.holdout)
        pop = engine.initial_population()

        # Initial population must have no holdout violators
        for g in pop:
            self.assertFalse(violates_holdout(g, self.holdout))

        # Next generation must also have no holdout violators
        evaluated = [(g, self.rng.random()) for g in pop]
        next_pop = engine.next_generation(evaluated, generation=1)
        for g in next_pop:
            self.assertFalse(violates_holdout(g, self.holdout))

    def test_build_holdout_eval_set(self):
        eco = PaymentEcosystem(num_customers=20, num_merchants=10, num_days=15)
        eco.generate_transactions()

        campaigns = build_holdout_eval_set(
            ecosystem=eco,
            holdout=self.holdout,
            n_campaigns=10,
            rng=self.rng,
        )

        self.assertGreater(len(campaigns), 0)
        for camp in campaigns:
            self.assertGreater(len(camp.transactions), 0)
            self.assertTrue(violates_holdout(camp.genome, self.holdout))

    def test_adaptive_mutation_rate(self):
        mut_engine = MutationEngine(self.rng, sigma=0.15)
        self.assertEqual(mut_engine.sigma, 0.15)

        # Saturation streak < 2 -> normal sigma
        mut_engine.set_saturation_streak(1)
        self.assertEqual(mut_engine.sigma, 0.15)

        # Saturation streak >= 2 -> boosted sigma
        mut_engine.set_saturation_streak(2)
        self.assertGreater(mut_engine.sigma, 0.15)

        mut_engine.set_saturation_streak(4)
        self.assertGreaterEqual(mut_engine.sigma, 0.25)


if __name__ == "__main__":
    unittest.main()
