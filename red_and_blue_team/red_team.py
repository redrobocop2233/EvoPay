"""
Red-team fraud agent, v0.4 - the evolutionary engine.

Earlier versions sampled one independent AttackStrategy per campaign and
scored it. This version turns that into an actual search: a population of
AttackGenomes is evaluated against the blue-team detector each generation,
scored on gain + evasion + novelty + scalability - cost, and the next
generation is bred from the survivors by mutation (with a share of fresh
random immigrants to keep the search from collapsing to one family).

Layers, matching the roles from the project concept:

  AttackGenome     - numeric vector the evolutionary algorithm manipulates
  GenomeCodec       - decodes a genome into an AttackStrategy (human-readable
                      patterns) and into a concrete Campaign via CampaignPlanner
  CampaignPlanner   - unchanged from v0.2/v0.3: strategy -> transactions
  HeuristicDetector - the v0.3 blue-team baseline being evolved against
  NoveltyEngine     - penalizes genomes too close to ones already tried
  FitnessEngine     - combines gain, evasion, novelty, scalability, cost
  StrategyMemory     - full record of every genome/campaign/result/fitness
  FailureAnalyzer   - which behavioral dimensions get caught vs. slip through
  EvolutionEngine   - selection + mutation + crossover + immigrants
  RedTeamController - runs the generational loop end to end

GenAI discovery/autopsy lives in the root ``genai/`` package. The Red Team
accepts GenAI-proposed genomes as ordinary candidates; GenAI never bypasses
realism validation or the fitness loop.
"""

import copy
import csv
import json
import random
from dataclasses import dataclass, asdict, field
from datetime import timedelta, datetime

from .ecosystem import PaymentEcosystem, new_id, MERCHANT_CATEGORIES, CURRENCY, DEVICE_TYPES, CHANNEL_MAP
from .blue_team import DetectorInterface, HeuristicDetector

GENE_NAMES = ["amount", "temporal", "device", "geographic", "merchant", "velocity", "coordination"]
ACTIVE_THRESHOLD = 0.65

# Blue's synthetic generator (data/generate_synthetic.py) only knows five
# named families. A single active dimension that clearly matches one gets
# that label, for compatibility with anything downstream that filters on
# attack_family. Anything else - including most evolved multi-dimension
# strategies - is legitimately novel and gets "emergent_..." rather than
# forced into a family it doesn't really match.
KNOWN_FAMILY_MAP = {
    "velocity": "velocity_abuse",
    "device": "account_takeover",
    "geographic": "geo_impossible_travel",
}


def attack_family_for(active_dimensions, strategy=None):
    if strategy is not None and strategy.amount_pattern.get("type") == "card_testing" \
            and active_dimensions == ["amount"]:
        return "card_testing"
    if len(active_dimensions) == 1 and active_dimensions[0] in KNOWN_FAMILY_MAP:
        return KNOWN_FAMILY_MAP[active_dimensions[0]]
    if not active_dimensions:
        return "legitimate"
    return "emergent_" + "_".join(sorted(active_dimensions))


# ---------------------------------------------------------------- genome ---

@dataclass
class AttackGenome:
    genome_id: str
    amount: float
    temporal: float
    device: float
    geographic: float
    merchant: float
    velocity: float
    coordination: float
    parent_id: str = None
    generation: int = 0
    mutation_summary: str = None

    def as_vector(self):
        return [getattr(self, g) for g in GENE_NAMES]


def random_genome(rng, generation=0, parent_id=None):
    return AttackGenome(
        genome_id=new_id("gen"),
        amount=rng.random(),
        temporal=rng.random(),
        device=rng.random(),
        geographic=rng.random(),
        merchant=rng.random(),
        velocity=rng.random(),
        coordination=rng.random(),
        parent_id=parent_id,
        generation=generation,
    )


class MutationEngine:
    def __init__(self, rng, sigma=0.15):
        self.rng = rng
        self.base_sigma = sigma
        self.sigma = sigma
        self._gene_variances = None  # set externally by EvolutionEngine

    def set_pressure(self, detection_rate: float | None):
        """Increase exploration when Blue is saturating, otherwise decay toward baseline."""
        if detection_rate is not None and detection_rate >= 0.95:
            self.sigma = min(0.30, self.sigma * 1.25)
        elif detection_rate is not None and detection_rate <= 0.50:
            self.sigma = max(self.base_sigma, self.sigma * 0.95)
        else:
            self.sigma = max(self.base_sigma, self.sigma * 0.99)

    def set_gene_variances(self, variances):
        """Accept per-gene variance from the population for biased mutation."""
        self._gene_variances = variances

    def mutate(self, genome, generation):
        child = copy.deepcopy(genome)
        child.genome_id = new_id("gen")
        child.parent_id = genome.genome_id
        child.generation = generation

        num_genes = self.rng.choice([1, 2, 3])

        # Diversity preservation: bias mutation toward low-variance genes
        # (the dimensions that have converged too much)
        if self._gene_variances is not None:
            inv_var = [1.0 / (v + 0.01) for v in self._gene_variances]
            total = sum(inv_var)
            weights = [w / total for w in inv_var]
            genes_to_mutate = []
            for _ in range(num_genes):
                gene = self.rng.choices(GENE_NAMES, weights=weights, k=1)[0]
                if gene not in genes_to_mutate:
                    genes_to_mutate.append(gene)
        else:
            genes_to_mutate = self.rng.sample(GENE_NAMES, k=num_genes)

        changed = []
        for gene in genes_to_mutate:
            before = getattr(child, gene)
            value = before + self.rng.gauss(0, self.sigma)
            value = min(1.0, max(0.0, value))
            setattr(child, gene, value)
            if abs(value - before) > 0.03:
                sign = "+" if value > before else "-"
                changed.append(f"{gene} {sign}{abs(value-before):.2f}")
        child.mutation_summary = ", ".join(changed) if changed else "no significant change"
        return child

    def crossover(self, genome_a, genome_b, generation):
        child = AttackGenome(
            genome_id=new_id("gen"),
            parent_id=f"{genome_a.genome_id}+{genome_b.genome_id}",
            generation=generation,
            **{g: (getattr(genome_a, g) + getattr(genome_b, g)) / 2 for g in GENE_NAMES},
        )
        child.mutation_summary = "crossover: " + genome_a.genome_id + " + " + genome_b.genome_id
        return child


# ------------------------------------------------------------ strategies ---

@dataclass
class AttackStrategy:
    strategy_id: str
    temporal_pattern: dict
    amount_pattern: dict
    device_pattern: dict
    geographic_pattern: dict
    merchant_pattern: dict
    velocity_pattern: dict
    coordination_pattern: dict
    parent_id: str = None
    generation: int = 0


class GenomeCodec:
    """Turns a continuous gene in [0, 1] into a discrete, parameterized
    pattern. Below ACTIVE_THRESHOLD every dimension decodes to "normal" -
    this is what keeps most of a random population close to legitimate
    behavior, with only some genomes expressing an active attack."""

    @staticmethod
    def decode(genome):
        return AttackStrategy(
            strategy_id=genome.genome_id,
            temporal_pattern=GenomeCodec._temporal(genome.temporal),
            amount_pattern=GenomeCodec._amount(genome.amount),
            device_pattern=GenomeCodec._device(genome.device),
            geographic_pattern=GenomeCodec._geographic(genome.geographic),
            merchant_pattern=GenomeCodec._merchant(genome.merchant),
            velocity_pattern=GenomeCodec._velocity(genome.velocity),
            coordination_pattern=GenomeCodec._coordination(genome.coordination),
            parent_id=genome.parent_id,
            generation=genome.generation,
        )

    @staticmethod
    def _scale(gene, lo_gene, hi_gene, lo_val, hi_val):
        progress = (gene - lo_gene) / (hi_gene - lo_gene)
        return lo_val + progress * (hi_val - lo_val)

    @staticmethod
    def _temporal(gene):
        if gene < ACTIVE_THRESHOLD:
            return {"type": "normal"}
        # was 0.4-1.0: every "active" temporal shift was already a strong
        # push toward midnight. Starting near 0 lets evolution find a
        # barely-noticeable shift instead of only ever blatant ones.
        magnitude = GenomeCodec._scale(gene, ACTIVE_THRESHOLD, 1.0, 0.05, 1.0)
        return {"type": "shift_to_offhours", "magnitude": round(magnitude, 2)}

    @staticmethod
    def _amount(gene):
        if gene < ACTIVE_THRESHOLD:
            return {"type": "normal"}
        if gene < 0.775:
            # was 0.5-2.0 (already a 50%+ jump at minimum)
            magnitude = GenomeCodec._scale(gene, ACTIVE_THRESHOLD, 0.775, 0.05, 2.0)
            duration = 3 + int(GenomeCodec._scale(gene, ACTIVE_THRESHOLD, 0.775, 0, 4))
            return {"type": "gradual_drift", "magnitude": round(magnitude, 2), "duration": duration}
        if gene < 0.90:
            # card_testing (IDENTIFY.md #11): several small, easily-dismissed
            # probe amounts to confirm a card is live, then one real attempt -
            # structurally different from both drift and spike, neither of
            # which has a "small probes then a real charge" shape
            probe_count = 2 + int(GenomeCodec._scale(gene, 0.775, 0.90, 0, 3))
            probe_amount = round(GenomeCodec._scale(gene, 0.775, 0.90, 0.5, 5.0), 2)
            final_magnitude = round(GenomeCodec._scale(gene, 0.775, 0.90, 2.0, 6.0), 2)
            return {"type": "card_testing", "probe_count": probe_count,
                    "probe_amount": probe_amount, "final_magnitude": final_magnitude}
        # was 3-12x (every abrupt_spike was at minimum a 3x normal transaction)
        magnitude = GenomeCodec._scale(gene, 0.90, 1.0, 1.1, 12)
        return {"type": "abrupt_spike", "magnitude": round(magnitude, 2)}

    @staticmethod
    def _device(gene):
        return {"type": "normal"} if gene < ACTIVE_THRESHOLD else {"type": "switch"}

    @staticmethod
    def _geographic(gene):
        if gene < ACTIVE_THRESHOLD:
            return {"type": "normal"}
        # was 0.2-0.8
        magnitude = GenomeCodec._scale(gene, ACTIVE_THRESHOLD, 1.0, 0.03, 0.8)
        return {"type": "distribution_shift", "magnitude": round(magnitude, 2)}

    @staticmethod
    def _merchant(gene):
        return {"type": "normal"} if gene < ACTIVE_THRESHOLD else {"type": "category_drift"}

    @staticmethod
    def _velocity(gene):
        if gene < ACTIVE_THRESHOLD:
            return {"type": "normal"}
        count = 2 + int(GenomeCodec._scale(gene, ACTIVE_THRESHOLD, 1.0, 0, 3))
        interval = 1 + int(GenomeCodec._scale(gene, ACTIVE_THRESHOLD, 1.0, 0, 9))
        return {"type": "burst", "count": count, "interval_minutes": interval}

    @staticmethod
    def _coordination(gene):
        # only meaningful for customers with more than one account - the
        # planner falls back to normal automatically if there's just one
        return {"type": "normal"} if gene < ACTIVE_THRESHOLD else {"type": "multi_account"}


class RealismValidator:
    """Rejects candidates that would be simulator artifacts, not plausible
    fraud - a hard cap on how extreme any single dimension can get, so
    evolution can't drift toward values the simulator would accept but no
    real payment network would (e.g. absurd amount multipliers)."""

    MAX_AMOUNT_MULTIPLIER = 15
    MAX_VELOCITY_COUNT = 6

    def is_realistic(self, strategy):
        amount = strategy.amount_pattern
        if amount["type"] == "abrupt_spike" and amount["magnitude"] > self.MAX_AMOUNT_MULTIPLIER:
            return False
        if amount["type"] == "gradual_drift" and amount["magnitude"] > 3.0:
            return False
        if strategy.velocity_pattern["type"] == "burst" and strategy.velocity_pattern["count"] > self.MAX_VELOCITY_COUNT:
            return False
        return True


# -------------------------------------------------------------- planner ---

@dataclass
class FraudTransaction:
    transaction_id: str
    campaign_id: str
    strategy_id: str
    customer_id: str
    account_id: str
    device_id: str
    merchant_id: str
    amount: float
    timestamp: str
    city: str
    merchant_category: str
    location_lat: float
    location_lon: float
    ip_address: str
    currency: str
    channel: str
    label: int
    attack_family: str = "unknown_synthetic"


def active_dimensions_of(strategy):
    dims = []
    if strategy.amount_pattern["type"] != "normal":
        dims.append("amount")
    if strategy.temporal_pattern["type"] != "normal":
        dims.append("temporal")
    if strategy.device_pattern["type"] != "normal":
        dims.append("device")
    if strategy.geographic_pattern["type"] != "normal":
        dims.append("geographic")
    if strategy.merchant_pattern["type"] != "normal":
        dims.append("merchant")
    if strategy.velocity_pattern["type"] != "normal":
        dims.append("velocity")
    if strategy.coordination_pattern["type"] != "normal":
        dims.append("coordination")
    return dims


class CampaignPlanner:
    def __init__(self, ecosystem, rng):
        self.eco = ecosystem
        self.rng = rng

    def build(self, customer, strategy):
        campaign_id = new_id("camp")
        accounts = self._accounts_for_campaign(customer, strategy)
        # (device_id, device_type) per account, so channel is always known -
        # including for a "switch" device that was never in the ecosystem
        device_info = {a.account_id: self._device_for_campaign(a, strategy) for a in accounts}

        num_txns = self._num_transactions(strategy)
        base_date = self._base_date()
        start_hour, start_minute = self._start_time(strategy)
        category = self._category_for_campaign(customer, strategy)
        family = attack_family_for(active_dimensions_of(strategy), strategy)

        transactions = []
        for i in range(num_txns):
            account = accounts[i % len(accounts)]
            city = self._city_for(customer, strategy)
            merchant = self._merchant_for(category, city)
            amount = self._amount_for(customer, strategy, i, num_txns)
            timestamp = self._timestamp_for(base_date, strategy, i, start_hour, start_minute)
            device_id, device_type = device_info[account.account_id]
            lat, lon = self.eco._jittered_coords(city)

            transactions.append(FraudTransaction(
                transaction_id=new_id("ftxn"),
                campaign_id=campaign_id,
                strategy_id=strategy.strategy_id,
                customer_id=customer.customer_id,
                account_id=account.account_id,
                device_id=device_id,
                merchant_id=merchant.merchant_id,
                amount=amount,
                timestamp=timestamp.isoformat(),
                city=city,
                merchant_category=category,
                location_lat=lat,
                location_lon=lon,
                ip_address=self.eco._random_ip(),
                currency=CURRENCY,
                channel=CHANNEL_MAP[device_type],
                label=1,
                attack_family=family,
            ))
        return campaign_id, transactions

    def _accounts_for_campaign(self, customer, strategy):
        all_accounts = [a for a in self.eco.accounts if a.account_id in customer.account_ids]
        if strategy.coordination_pattern["type"] == "multi_account" and len(all_accounts) >= 2:
            return all_accounts
        return [self.rng.choice(all_accounts)]

    def _num_transactions(self, strategy):
        if strategy.amount_pattern["type"] == "gradual_drift":
            return strategy.amount_pattern["duration"]
        if strategy.amount_pattern["type"] == "card_testing":
            return strategy.amount_pattern["probe_count"] + 1  # probes + one real charge
        if strategy.velocity_pattern["type"] == "burst":
            return strategy.velocity_pattern["count"]
        if strategy.coordination_pattern["type"] == "multi_account":
            return 2
        return 1

    def _base_date(self):
        start = datetime(2026, 4, 1)
        return start + timedelta(days=self.rng.randint(0, self.eco.num_days - 1))

    def _device_for_campaign(self, account, strategy):
        if strategy.device_pattern["type"] == "switch":
            return new_id("dev"), self.rng.choice(DEVICE_TYPES)
        device_id = self.rng.choice(account.device_ids)
        return device_id, self.eco._device_type(device_id)

    def _category_for_campaign(self, customer, strategy):
        if strategy.merchant_pattern["type"] == "category_drift":
            unused = [c for c in MERCHANT_CATEGORIES if c not in customer.favorite_categories]
            return self.rng.choice(unused) if unused else self.rng.choice(customer.favorite_categories)
        return self.rng.choice(customer.favorite_categories)

    def _city_for(self, customer, strategy):
        if strategy.geographic_pattern["type"] != "distribution_shift":
            return self.eco._sample_city(customer.location_weights)
        return self._shifted_city(customer.location_weights, strategy.geographic_pattern["magnitude"])

    def _shifted_city(self, location_weights, magnitude):
        cities = list(location_weights.keys())
        normal_probs = list(location_weights.values())
        rare_probs = [1.0 - p for p in normal_probs]
        rare_total = sum(rare_probs)
        rare_probs = [p / rare_total for p in rare_probs] if rare_total > 0 else normal_probs
        blended = [(1 - magnitude) * n + magnitude * r for n, r in zip(normal_probs, rare_probs)]
        return self.rng.choices(cities, weights=blended)[0]

    def _merchant_for(self, category, city):
        pool = [m for m in self.eco.merchants if m.category == category and m.city == city]
        if not pool:
            pool = [m for m in self.eco.merchants if m.category == category] or self.eco.merchants
        return self.rng.choice(pool)

    def _amount_for(self, customer, strategy, index, num_txns):
        pattern = strategy.amount_pattern
        if pattern["type"] == "gradual_drift":
            progress = index / max(1, num_txns - 1)
            base = customer.avg_amount * (1 + pattern["magnitude"] * progress)
            return round(max(10.0, self.rng.gauss(base, customer.amount_std * 0.5)), 2)
        if pattern["type"] == "abrupt_spike":
            return round(customer.avg_amount * pattern["magnitude"], 2)
        if pattern["type"] == "card_testing":
            is_final_charge = (index == num_txns - 1)
            if is_final_charge:
                return round(customer.avg_amount * pattern["final_magnitude"], 2)
            return round(max(0.5, self.rng.gauss(pattern["probe_amount"], 0.5)), 2)
        return round(max(10.0, self.rng.gauss(customer.avg_amount, customer.amount_std)), 2)

    def _start_time(self, strategy):
        if strategy.temporal_pattern["type"] == "shift_to_offhours":
            hour = self.rng.choice([0, 1, 2, 3, 4])
        else:
            hour = self.rng.randint(0, 23)
        minute = self.rng.randint(0, 59)
        return hour, minute

    def _timestamp_for(self, base_date, strategy, index, start_hour, start_minute):
        date = base_date + timedelta(days=index) if strategy.amount_pattern["type"] == "gradual_drift" else base_date
        ts = date.replace(hour=start_hour, minute=start_minute)
        if strategy.velocity_pattern["type"] == "burst":
            ts += timedelta(minutes=index * strategy.velocity_pattern["interval_minutes"])
        if strategy.amount_pattern["type"] == "card_testing" and index > 0:
            # probes fire in rapid succession (seconds to a couple minutes
            # apart) - that tight clustering around one start time is the
            # actual signature of card testing, not something the separate
            # velocity gene should have to coincidentally also be active for
            ts = date.replace(hour=start_hour, minute=start_minute) + timedelta(
                seconds=index * self.rng.randint(20, 90))
        return ts


# ----------------------------------------------------- novelty & fitness ---

class NoveltyEngine:
    """Distance to the nearest genomes already tried, in 7-dimensional gene
    space. Prevents the population from converging on many near-duplicates
    of one winning strategy and calling that "diversity"."""

    def __init__(self, k=5, max_archive=800):
        self.archive = []
        self.k = k
        self.max_archive = max_archive

    def score(self, genome):
        if not self.archive:
            return 1.0
        vector = genome.as_vector()
        distances = sorted(self._distance(vector, v) for v in self.archive)
        nearest = distances[:self.k]
        return sum(nearest) / len(nearest)

    def add(self, genome):
        self.archive.append(genome.as_vector())
        if len(self.archive) > self.max_archive:
            self.archive.pop(0)

    def _distance(self, a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class FitnessEngine:
    WEIGHTS = {"gain": 0.20, "evasion": 0.45, "novelty": 0.20, "scalability": 0.10, "cost": 0.05}
    MAX_NOVELTY_DISTANCE = 2.6  # ~sqrt(7), the diagonal of the unit gene-space cube

    def score(self, gain, num_txns, customer_avg_amount, detection_result, novelty_raw):
        # was /3: any transaction ~3x normal already saturated gain_score at
        # 1.0, so fitness couldn't tell a 3x attack from a 12x one apart -
        # /8 keeps some headroom so gain differences still matter
        gain_score = min(1.0, gain / max(1.0, customer_avg_amount * num_txns * 8))
        evasion_score = 1.0 - detection_result.risk_score
        novelty_score = min(1.0, novelty_raw / self.MAX_NOVELTY_DISTANCE)
        scalability_score = min(1.0, num_txns / 5)
        cost_score = min(1.0, num_txns / 10)

        fitness = (
            self.WEIGHTS["gain"] * gain_score
            + self.WEIGHTS["evasion"] * evasion_score
            + self.WEIGHTS["novelty"] * novelty_score
            + self.WEIGHTS["scalability"] * scalability_score
            - self.WEIGHTS["cost"] * cost_score
        )
        components = {
            "gain_score": round(gain_score, 3),
            "evasion_score": round(evasion_score, 3),
            "novelty_score": round(novelty_score, 3),
            "scalability_score": round(scalability_score, 3),
            "cost_score": round(cost_score, 3),
        }
        return round(fitness, 4), components


# ------------------------------------------------------------- memory ---

@dataclass
class MemoryRecord:
    genome_id: str
    parent_id: str
    generation: int
    campaign_id: str
    customer_id: str
    num_transactions: int
    gain: float
    risk_score: float
    detected: bool
    reason_codes: list
    fitness: float
    gain_score: float
    evasion_score: float
    novelty_score: float
    active_dimensions: list  # which strategy dimensions were non-"normal"
    genome_vector: list = field(default_factory=list)
    attack_family: str = "unknown"
    detection_probability: float = 0.0
    attack_success: float = 0.0
    parent_campaign_id: str = None
    mutation_summary: str = None


class StrategyMemory:
    def __init__(self):
        self.records = []

    def record(self, entry):
        self.records.append(entry)

    def top(self, n=10):
        return sorted(self.records, key=lambda r: r.fitness, reverse=True)[:n]

    def to_csv(self, path):
        if not self.records:
            return
        rows = [asdict(r) for r in self.records]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                row = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()}
                writer.writerow(row)


class FailureAnalyzer:
    """Per behavioral dimension: how often it was active, and of those, how
    often the campaign still got caught. Low detection rate on a dimension
    that's been tried a lot is exactly the "detector blind spot" signal the
    evolutionary search should end up exploiting more over time."""

    def analyze(self, memory: StrategyMemory):
        summary = {dim: {"active": 0, "detected": 0} for dim in GENE_NAMES}
        for record in memory.records:
            for dim in record.active_dimensions:
                summary[dim]["active"] += 1
                if record.detected:
                    summary[dim]["detected"] += 1

        report = {}
        for dim, counts in summary.items():
            if counts["active"] == 0:
                report[dim] = {"active": 0, "detected": 0, "detection_rate": None}
            else:
                report[dim] = {
                    "active": counts["active"],
                    "detected": counts["detected"],
                    "detection_rate": round(counts["detected"] / counts["active"], 3),
                }
        return report


# ---------------------------------------------------------- evolution ---

class EvolutionEngine:
    def __init__(self, rng, population_size=30, survival_fraction=0.3, immigrant_fraction=0.15):
        self.rng = rng
        self.population_size = population_size
        self.num_survivors = max(2, int(population_size * survival_fraction))
        self.num_immigrants = max(1, int(population_size * immigrant_fraction))
        self.mutation_engine = MutationEngine(rng)

    def initial_population(self):
        return [random_genome(self.rng, generation=0) for _ in range(self.population_size)]

    def next_generation(self, evaluated, generation):
        # evaluated: list of (genome, fitness)
        ranked = sorted(evaluated, key=lambda pair: pair[1], reverse=True)

        # Adaptive mutation pressure: if the detector catches almost everything,
        # increase exploration so Red does not stagnate at a saturated defense.
        valid = [g for g, f in evaluated if f >= 0]
        if valid:
            # Fitness is not detection, so derive pressure from detector outcomes
            # stored by the caller only when available via the previous generation.
            pass

        # Section 3 — Family-balanced selection: cap how many survivors
        # can come from the same attack family so one high-fitness family
        # can't dominate the entire next generation.
        survivors = self._family_balanced_select(ranked)

        # Update gene variances for variance-biased mutation
        vectors = [g.as_vector() for g, _ in evaluated if _ >= 0]
        if vectors:
            import numpy as _np
            arr = _np.array(vectors)
            gene_variances = [float(arr[:, i].var()) for i in range(len(GENE_NAMES))]
            self.mutation_engine.set_gene_variances(gene_variances)

        children = []
        num_children = self.population_size - len(survivors) - self.num_immigrants
        for _ in range(num_children):
            if len(survivors) >= 2 and self.rng.random() < 0.3:
                parent_a, parent_b = self.rng.sample(survivors, 2)
                children.append(self.mutation_engine.crossover(parent_a, parent_b, generation))
            else:
                parent = self.rng.choice(survivors)
                children.append(self.mutation_engine.mutate(parent, generation))

        immigrants = [random_genome(self.rng, generation=generation) for _ in range(self.num_immigrants)]
        return survivors + children + immigrants

    def _family_balanced_select(self, ranked):
        """Select survivors with a per-family cap to preserve diversity."""
        # Determine families for each genome
        family_for = {}
        for genome, fitness in ranked:
            strategy = GenomeCodec.decode(genome)
            active = active_dimensions_of(strategy)
            family = attack_family_for(active, strategy)
            family_for[genome.genome_id] = family

        all_families = set(family_for.values())
        max_per_family = max(2, self.num_survivors // max(1, len(all_families)))

        survivors = []
        family_counts = {}
        for genome, fitness in ranked:
            if len(survivors) >= self.num_survivors:
                break
            family = family_for.get(genome.genome_id, "unknown")
            if family_counts.get(family, 0) < max_per_family:
                survivors.append(genome)
                family_counts[family] = family_counts.get(family, 0) + 1

        # If we didn't fill enough survivors (all families hit cap), fill remainder
        if len(survivors) < self.num_survivors:
            for genome, fitness in ranked:
                if genome not in survivors and len(survivors) < self.num_survivors:
                    survivors.append(genome)

        return survivors


# ------------------------------------------------------------ controller ---

class RedTeamController:
    def __init__(self, ecosystem: PaymentEcosystem, detector: DetectorInterface = None, seed=7,
                 population_size=30):
        self.eco = ecosystem
        self.rng = random.Random(seed)
        self.detector = detector or HeuristicDetector()
        self.planner = CampaignPlanner(ecosystem, self.rng)
        self.validator = RealismValidator()
        self.novelty_engine = NoveltyEngine()
        self.fitness_engine = FitnessEngine()
        self.evolution = EvolutionEngine(self.rng, population_size=population_size)
        self.memory = StrategyMemory()
        self.fraud_transactions = []
        self.fraud_transactions_by_campaign = {}  # campaign_id -> [txns]
        self.generation_stats = []
        self.population = None
        self.next_generation_number = 0
        self.genome_to_campaign = {}

    def _inject_seed_genomes(self, population, seed_genomes):
        """Inject externally proposed genomes without discarding the running population."""
        if not seed_genomes:
            return population
        population = list(population)
        for genome in seed_genomes[: self.evolution.population_size]:
            if len(population) < self.evolution.population_size:
                population.append(genome)
            else:
                population[self.rng.randrange(len(population))] = genome
        return population[: self.evolution.population_size]

    def evolve(self, generations=10, seed_genomes=None, holdout=None):
        """seed_genomes: optional list of AttackGenome (e.g. from
        genai.discoverer.hypothesis_to_genome) mixed into generation 0
        alongside the usual random population. They compete on fitness like
        anything else - being GenAI-proposed doesn't exempt a genome from
        the realism validator or from losing out to a stronger mutation."""
        if self.population is None:
            population = self.evolution.initial_population()
        else:
            population = list(self.population)
        population = self._inject_seed_genomes(population, seed_genomes)

        for _ in range(generations):
            generation = self.next_generation_number
            evaluated = []
            gen_detected = 0
            gen_fitness_total = 0.0
            gen_risk_total = 0.0
            gen_evaluated = 0

            for genome in population:
                strategy = GenomeCodec.decode(genome)
                if holdout is not None:
                    from eval.holdout import violates_holdout
                    if violates_holdout(strategy, holdout):
                        evaluated.append((genome, -1.0))
                        continue
                if not self.validator.is_realistic(strategy):
                    # rejected candidates don't compete this generation but are
                    # still replaced next generation via mutation of survivors
                    evaluated.append((genome, -1.0))
                    continue

                customer = self.rng.choice(self.eco.customers)
                campaign_id, transactions = self.planner.build(customer, strategy)
                detection_result = self.detector.evaluate(transactions, self.eco)

                gain = sum(t.amount for t in transactions)
                novelty_raw = self.novelty_engine.score(genome)
                fitness, components = self.fitness_engine.score(
                    gain, len(transactions), customer.avg_amount, detection_result, novelty_raw)

                self.novelty_engine.add(genome)
                self.fraud_transactions.extend(transactions)
                self.fraud_transactions_by_campaign[campaign_id] = transactions

                active_dims = active_dimensions_of(strategy)
                detection_probability = detection_result.risk_score
                attack_success_val = 1.0 - detection_probability
                self.memory.record(MemoryRecord(
                    genome_id=genome.genome_id,
                    parent_id=genome.parent_id,
                    generation=genome.generation,
                    campaign_id=campaign_id,
                    customer_id=customer.customer_id,
                    num_transactions=len(transactions),
                    gain=round(gain, 2),
                    risk_score=detection_result.risk_score,
                    detected=detection_result.detected,
                    reason_codes=detection_result.reason_codes,
                    fitness=fitness,
                    gain_score=components["gain_score"],
                    evasion_score=components["evasion_score"],
                    novelty_score=components["novelty_score"],
                    active_dimensions=active_dims,
                    genome_vector=genome.as_vector(),
                    attack_family=attack_family_for(active_dims, strategy),
                    detection_probability=detection_probability,
                    attack_success=attack_success_val,
                    parent_campaign_id=self._parent_campaign_id(genome),
                    mutation_summary=genome.mutation_summary,
                ))

                self.genome_to_campaign[genome.genome_id] = campaign_id
                evaluated.append((genome, fitness))
                gen_evaluated += 1
                gen_fitness_total += fitness
                gen_risk_total += detection_result.risk_score
                if detection_result.detected:
                    gen_detected += 1

            if gen_evaluated > 0:
                self.generation_stats.append({
                    "generation": generation,
                    "avg_fitness": round(gen_fitness_total / gen_evaluated, 4),
                    "avg_risk": round(gen_risk_total / gen_evaluated, 3),
                    "detection_rate": round(gen_detected / gen_evaluated, 3),
                    "attack_success_rate": round(1 - gen_detected / gen_evaluated, 3),
                    "evaluated": gen_evaluated,
                })

            if gen_evaluated > 0:
                self.evolution.mutation_engine.set_pressure(gen_detected / gen_evaluated)
            population = self.evolution.next_generation(evaluated, generation + 1)
            self.next_generation_number = generation + 1

        self.population = population
        return self.memory

    def _parent_campaign_id(self, genome):
        if not genome.parent_id:
            return None
        # Crossover stores two parent genome IDs separated by '+'. Use the
        # first parent for a simple tree while retaining both in mutation_summary.
        parent_genome_id = genome.parent_id.split("+")[0]
        return self.genome_to_campaign.get(parent_genome_id)

    def to_csv(self, out_dir="."):
        self.memory.to_csv(f"{out_dir}/strategy_memory.csv")
        self._write_csv(f"{out_dir}/fraud_transactions.csv", self.fraud_transactions)
        self._write_generation_stats(f"{out_dir}/generation_stats.csv")

    def _write_csv(self, path, records):
        if not records:
            return
        rows = [asdict(r) for r in records]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_generation_stats(self, path):
        if not self.generation_stats:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.generation_stats[0].keys()))
            writer.writeheader()
            for row in self.generation_stats:
                writer.writerow(row)


if __name__ == "__main__":
    eco = PaymentEcosystem(num_customers=200, num_merchants=60, num_days=90)
    eco.generate_transactions()

    controller = RedTeamController(eco, population_size=30)
    memory = controller.evolve(generations=15)
    controller.to_csv(out_dir=".")

    print("generation | avg fitness | avg risk | detection rate")
    for stat in controller.generation_stats:
        print(f"{stat['generation']:>10} | {stat['avg_fitness']:>11} | {stat['avg_risk']:>8} | {stat['detection_rate']}")

    print()
    print("top 5 strategies by fitness:")
    for record in memory.top(5):
        print(f"  gen {record.generation} | fitness {record.fitness} | gain {record.gain} | "
              f"risk {record.risk_score} | detected {record.detected} | dims {record.active_dimensions}")

    print()
    print("detection rate by active dimension (blind-spot signal):")
    analyzer = FailureAnalyzer()
    for dim, stats in analyzer.analyze(memory).items():
        print(f"  {dim:12} active={stats['active']:>4} detection_rate={stats['detection_rate']}")
