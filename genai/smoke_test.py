from dotenv import load_dotenv
load_dotenv(override=True)

"""
Live smoke test for genai/ - requires GEMINI_API_KEY.

Everything in genai/ was tested against a mocked LLM client in development
(schema validation, genome round-trip math, boundary cases, evolution
integration) since no API key was available in that environment. This
script is the one thing that couldn't be verified there: the actual live
call to the Gemini model and whether its real output validates against the strict
schemas in genai/schemas.py.

Run: python -m genai.smoke_test
"""

import os
import sys

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set - export it first:")
        print("  export GEMINI_API_KEY=sk-...")
        sys.exit(1)

    from genai.client import GenAIClient
    from genai.discoverer import ThreatDiscoverer, hypothesis_to_genome
    from genai.analyst import AttackAnalyst
    import random

    print("1. Threat discovery (live Gemini call)...")
    client = GenAIClient()
    discoverer = ThreatDiscoverer(client=client)
    simulatable, research_only = discoverer.discover(
        n=5, known_families=["card_testing", "account_takeover", "velocity_abuse"],
        weak_dimensions=["geographic"],
    )
    print(f"   {len(simulatable)} transaction_simulatable, {len(research_only)} research_only")
    for h in simulatable:
        print(f"   [SIM] {h.name} ({h.family})")
    for h in research_only:
        print(f"   [RESEARCH] {h.name} ({h.family})")

    if not simulatable:
        print("   No simulatable hypotheses returned - can't test genome conversion.")
    else:
        print()
        print("2. Converting first simulatable hypothesis to a genome...")
        genome = hypothesis_to_genome(simulatable[0], random.Random(0))
        print(f"   genome: amount={genome.amount:.3f} temporal={genome.temporal:.3f} "
              f"device={genome.device:.3f} geographic={genome.geographic:.3f}")

    print()
    print("3. Attack autopsy (live call)...")
    analyst = AttackAnalyst(client=client)
    autopsy = analyst.analyze(
        strategy_id="smoke_test_strategy",
        strategy_summary={"amount_pattern": {"type": "abrupt_spike", "magnitude": 8.0},
                          "device_pattern": {"type": "switch"}},
        detection_result={"risk_score": 0.91, "detected": True,
                          "reason_codes": ["amount_deviation", "unknown_device"]},
    )
    print(f"   explanation: {autopsy.explanation}")
    print(f"   weakest_signal: {autopsy.weakest_signal}")
    print(f"   recommended_mutations: {[m.dimension + ':' + m.direction for m in autopsy.recommended_mutations]}")

    print()
    print("Smoke test passed - live API calls validated against the strict schemas.")


if __name__ == "__main__":
    main()
