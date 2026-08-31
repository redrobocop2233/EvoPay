"""GenAI Threat Discovery layer for evo-pay.

Optional: everything else in this project (ecosystem, red_team, blue_team,
API, dashboard) works with zero GenAI dependency. This package adds an
LLM-driven layer on top, gated behind OPENAI_API_KEY:

    genai.discoverer.ThreatDiscoverer  - propose new attack hypotheses,
        grounded in IDENTIFY.md, convertible into real AttackGenomes
    genai.analyst.AttackAnalyst        - explain a detection result and
        suggest a mutation direction, after the fact - never part of the
        actual detection decision
"""
