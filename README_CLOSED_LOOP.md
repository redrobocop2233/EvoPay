# EVO-PAY: Adaptive GenAI Red Team / Blue Team

EVO-PAY is a closed-loop synthetic payment-security laboratory:

1. **Identify** — Gemini proposes emerging behavioral fraud hypotheses.
2. **Generate** — hypotheses are compiled into a bounded Red Team genome and
   evolved into realistic synthetic campaigns.
3. **Defend** — the Blue detector scores the campaigns.
4. **Autopsy** — Gemini explains completed detections and proposes bounded
   mutation directions.
5. **Adapt** — Red explores the proposed directions; optionally, Blue retrains
   on the accumulated adversarial feed.

The key design rule is separation of authority:

> **Gemini proposes. Red simulates. Blue decides. Gemini explains.**

See `INTEGRATION.md` for the runnable commands and output artifacts.
