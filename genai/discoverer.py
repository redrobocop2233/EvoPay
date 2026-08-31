"""GenAI threat discovery -> validated hypotheses -> Red genomes."""
from __future__ import annotations

import json
import random

from .schemas import AttackHypothesis
from red_team.red_team import AttackGenome, ACTIVE_THRESHOLD, GenomeCodec


DISCOVERY_SYSTEM_PROMPT = """You are EVO-PAY's defensive GenAI Threat Discovery Agent.
Generate plausible emerging payment-fraud behavioral hypotheses for a synthetic payment simulator.
Return only JSON matching the requested schema. Do not provide real-world fraud instructions,
credentials, phishing content, malware, or targeting guidance. Focus on abstract transaction behavior.
Mark research_only when a threat needs unsupported modalities such as voice, video, identity documents,
or communications. Transaction-simulatable hypotheses must use the supplied pattern vocabulary."""


class ThreatDiscoverer:
    """Generate validated attack hypotheses. The LLM never writes transaction rows."""

    def __init__(self, client):
        self.client = client

    def discover(self, n=5, known_families=None, weak_dimensions=None):
        prompt = {
            "task": "discover novel or underrepresented payment-fraud behavioral strategies",
            "requested_hypotheses": int(n),
            "known_attack_families": known_families or [],
            "blue_team_weak_dimensions": weak_dimensions or [],
            "available_dimensions": {
                "temporal": ["normal", "shift_to_offhours"],
                "amount": ["normal", "gradual_drift", "abrupt_spike", "card_testing"],
                "device": ["normal", "switch"],
                "geographic": ["normal", "distribution_shift"],
                "merchant": ["normal", "category_drift"],
                "velocity": ["normal", "burst"],
                "coordination": ["normal", "multi_account"],
            },
            "requirements": [
                "prefer combinations of behavioral weaknesses",
                "avoid simply reproducing known families",
                "keep simulatable hypotheses schema-compatible",
            ],
        }
        hypotheses = self.client.complete_json_list(
            DISCOVERY_SYSTEM_PROMPT,
            json.dumps(prompt, indent=2),
            AttackHypothesis,
            4000,
        )
        return (
            [h for h in hypotheses if h.modality == "transaction_simulatable"],
            [h for h in hypotheses if h.modality == "research_only"],
        )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scale(value, src_lo, src_hi, dst_lo, dst_hi):
    if src_hi == src_lo:
        return dst_lo
    ratio = (float(value) - src_lo) / (src_hi - src_lo)
    ratio = max(0.0, min(1.0, ratio))
    return dst_lo + ratio * (dst_hi - dst_lo)


def hypothesis_to_genome(hypothesis: AttackHypothesis, rng: random.Random):
    """Compile semantic GenAI patterns into the Red Team's [0,1]^7 genome."""
    amount_p = hypothesis.amount_pattern
    if not amount_p or amount_p.type == "normal":
        amount = 0.0
    elif amount_p.type == "gradual_drift":
        amount = _scale(amount_p.magnitude or 0.05, 0.05, 2.0, ACTIVE_THRESHOLD, 0.775)
    elif amount_p.type == "card_testing":
        amount = _scale(amount_p.final_magnitude or 2.0, 2.0, 6.0, 0.775, 0.90)
    elif amount_p.type == "abrupt_spike":
        amount = _scale(amount_p.magnitude or 1.1, 1.1, 12.0, 0.90, 1.0)
    else:
        amount = 0.0

    temporal_p = hypothesis.temporal_pattern
    temporal = 0.0 if not temporal_p or temporal_p.type == "normal" else _scale(temporal_p.magnitude or 0.05, 0.05, 1.0, ACTIVE_THRESHOLD, 1.0)

    device = 0.0 if not hypothesis.device_pattern or hypothesis.device_pattern.type == "normal" else 0.80

    geo_p = hypothesis.geographic_pattern
    geographic = 0.0 if not geo_p or geo_p.type == "normal" else _scale(geo_p.magnitude or 0.03, 0.03, 0.8, ACTIVE_THRESHOLD, 1.0)

    merchant = 0.0 if not hypothesis.merchant_pattern or hypothesis.merchant_pattern.type == "normal" else 0.80

    velocity_p = hypothesis.velocity_pattern
    velocity = 0.0 if not velocity_p or velocity_p.type == "normal" else _scale(velocity_p.count or 2, 2, 5, ACTIVE_THRESHOLD, 1.0)

    coordination = 0.0 if not hypothesis.coordination_pattern or hypothesis.coordination_pattern.type == "normal" else 0.80

    return AttackGenome(
        genome_id=f"genai_{rng.getrandbits(64):016x}",
        amount=_clamp01(amount),
        temporal=_clamp01(temporal),
        device=_clamp01(device),
        geographic=_clamp01(geographic),
        merchant=_clamp01(merchant),
        velocity=_clamp01(velocity),
        coordination=_clamp01(coordination),
        generation=0,
    )
