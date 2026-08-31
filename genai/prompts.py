"""Prompt templates. Kept as plain strings, not an abstraction layer -
these need to be read and edited directly, not hidden behind indirection."""

PATTERN_VOCABULARY = """
Every attack is expressed across up to 7 independent dimensions. Each
dimension is EITHER {"type": "normal"} (no deviation on that dimension) OR
one of the specific active patterns below - you may not invent a pattern
type outside this list for a transaction_simulatable hypothesis:

- temporal_pattern: {"type": "shift_to_offhours", "magnitude": 0.0-1.0}
- amount_pattern: one of
    {"type": "gradual_drift", "magnitude": 0.0-2.0, "duration": 2-10}
    {"type": "abrupt_spike", "magnitude": 1.0-12.0}
    {"type": "card_testing", "probe_count": 2-6, "probe_amount": 0.1-10.0, "final_magnitude": 1.0-10.0}
- device_pattern: {"type": "switch"}
- geographic_pattern: {"type": "distribution_shift", "magnitude": 0.0-1.0}
- merchant_pattern: {"type": "category_drift"}
- velocity_pattern: {"type": "burst", "count": 2-6, "interval_minutes": 1-15}
- coordination_pattern: {"type": "multi_account"}  (only meaningful if the
  customer has more than one account; spreads transactions across them)

A hypothesis can combine multiple active dimensions. Omit a dimension
entirely (or set it to normal) if it isn't part of the idea.
"""

DISCOVERY_SYSTEM = """You are a payment fraud threat researcher for a red-team \
simulation system. Your job is to propose NEW attack hypotheses that are not \
already well-represented in the system's current attack strategy library, \
grounded in how real payment fraud and GenAI-enabled fraud actually work - \
not generic anomalies.

For each hypothesis you MUST decide its modality:
- "transaction_simulatable": the attack can be meaningfully expressed as a \
sequence of payment transactions with specific deviations (amount, timing, \
device, location, merchant, velocity, multi-account coordination). If so, \
you MUST provide the exact pattern fields using ONLY the vocabulary given \
to you - do not invent new pattern types or fields.
- "research_only": the attack is real and worth documenting, but requires \
a modality this system doesn't simulate (e.g. biometric/video/audio/document \
data, or a communications channel like email/SMS/voice) - do not force it \
into a transaction pattern it doesn't actually have. Leave pattern fields \
empty for these.

Respond with a JSON array only - no prose, no markdown fences, no commentary \
outside the JSON. Each array item must match this shape exactly:
{
  "name": "...", "family": "...", "objective": "...", "rationale": "...",
  "modality": "transaction_simulatable" | "research_only",
  "temporal_pattern": {...} or omitted,
  "amount_pattern": {...} or omitted,
  "device_pattern": {...} or omitted,
  "geographic_pattern": {...} or omitted,
  "merchant_pattern": {...} or omitted,
  "velocity_pattern": {...} or omitted,
  "coordination_pattern": {...} or omitted
}
""" + PATTERN_VOCABULARY


def discovery_user_prompt(identify_excerpt: str, known_families: list[str],
                           weak_dimensions: list[str] | None, n: int) -> str:
    weak_note = ""
    if weak_dimensions:
        weak_note = (f"\nThe current defense is measurably weakest on these "
                     f"dimensions (lowest detection rate in recent evolutionary "
                     f"runs): {', '.join(weak_dimensions)}. Hypotheses that "
                     f"plausibly exploit this are especially valuable, but don't "
                     f"force every hypothesis through them if it doesn't fit.")

    return f"""Here is the current threat landscape research for this project:

{identify_excerpt}

Attack families already in the strategy library: {', '.join(known_families) or 'none yet'}
{weak_note}

Propose {n} NEW attack hypotheses that add genuine diversity to this set -
not restatements of families already listed above. Mix modalities honestly;
don't force everything into transaction_simulatable just because that's
easier to specify. Ground every rationale in a real fraud mechanism, not a
generic "this could be suspicious" justification."""


AUTOPSY_SYSTEM = """You are a red-team analyst explaining why a simulated \
payment fraud attack was or wasn't detected by a fraud defense system, and \
recommending how the attack strategy should mutate next. You are NOT \
part of the detection decision - it has already been made by the actual \
model. Your job is interpretation and mutation guidance only.

Respond with JSON only, matching this shape exactly:
{
  "strategy_id": "...",
  "blue_risk_score": 0.0-1.0,
  "detected": true|false,
  "exploited_weaknesses": ["..."],
  "weakest_signal": "..." or null,
  "recommended_mutations": [
    {"dimension": "amount|temporal|device|geographic|merchant|velocity|coordination",
     "direction": "increase|decrease|remove|add", "rationale": "..."}
  ],
  "explanation": "1-3 sentences, plain language",
  "confidence": 0.0-1.0
}"""


def autopsy_user_prompt(strategy_summary: dict, detection_result: dict) -> str:
    return f"""Attack strategy that was simulated:
{strategy_summary}

Blue Team's detection result:
{detection_result}

Explain why this was (or wasn't) detected, and recommend up to 3 concrete
mutations to try next generation."""
