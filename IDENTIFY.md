# Identify: the GenAI payment fraud landscape

Grounded in FBI/FinCEN advisories, Federal Reserve/Boston Fed research, and
2026 industry threat reports (Deloitte, Experian, Group-IB, Shufti, Zyphe,
FSSCC). Sources cited inline. This is the "Identify" pillar: breadth first,
then an honest map of which of these our Generate/Defend pipeline actually
simulates today versus which require a different system entirely.

## A. Identity & onboarding fraud

1. **Synthetic identity fraud** - a real SSN/PII fragment combined with a
   fabricated name, DOB, and GenAI-generated face/history to build a credit
   profile that matures before being drained. No real victim exists to
   report it, so detection lags. Fastest-growing detected attack class on
   at least one major identity network (~31% YoY). [Federal Reserve Bank of
   Boston 2025; fedpaymentsimprovement.org SIF toolkit; Zyphe 2026]
2. **Deepfake liveness/selfie-check bypass** - a generated face that blinks,
   turns, lip-syncs, and responds to a liveness prompt in real time, or an
   injection attack replacing the camera feed entirely. Injection attacks
   reportedly rose ~200% (2023) and deepfake biometric attempts ~58% by 2026
   estimates. [Shufti Identity Fraud Index 2026; Gartner via same]
3. **Document deepfakes** - AI-forged or face-swapped ID documents/video
   submitted at KYC. NFC chip verification is one of the few deterministic
   counters, since the issuing authority never signed the assembled fake.
   [Zyphe 2026]

## B. Authentication & account takeover

4. **Deepfake voice social engineering against call centers/help desks** -
   cloned voice used to pass agent verification and request a password
   reset, card reissue, or transfer. [FSSCC AI-IA Workstream Mitigations,
   Tactic 2]
5. **Real-time deepfake video during step-up authentication** - live
   video-call deepfake to defeat human-in-the-loop verification.
   [FSSCC Tactic 9; Shufti]
6. **Agentic account takeover** - an AI agent given stolen credentials
   navigates the account UI like a human (mimicking mouse/typing behavior),
   defeating bot-detection, then moves funds to a mule account. Group-IB
   ran this as a proof-of-concept with a public agent framework and stolen
   credentials, unmodified. [Group-IB High-Tech Crime Trends Report 2026]

## C. Social engineering / payment authorization fraud

7. **AI-generated phishing/BEC** - near-flawless, personalized emails
   impersonating a CEO or vendor, often scraped from public data. One 2024
   industry report cites a 118% rise in AI-driven phishing/deepfake
   activity. [arXiv 2504.21574, Global Survey of GenAI in FI]
8. **Deepfake voice/video CEO fraud** - cloned executive voice/video
   authorizing a wire transfer; ~26% of targeted executives in one study
   said the attacker's goal was an unauthorized transfer. [same source]
9. **Multi-vector campaigns** - a single operator orchestrating email +
   voice (vishing, +442% in 2024) + SMS from one LLM-driven campaign,
   spanning ~41% of campaigns by one estimate. [arXiv 2507.12185]
10. **Romance/relationship scams ("pig butchering") run by LLM chat agents**
    at scale, sustaining long-running personalized conversations that would
    previously have required a human operator per victim. [arXiv 2412.15423,
    cited in "When AI Agents Collude Online"]

## D. Transaction-level fraud (card-present & card-not-present)

11. **Automated card testing** - bots probing stolen card numbers with
    small, easily-missed purchases to find which cards are still active,
    before using the confirmed ones for larger fraud. [Group-IB 2026]
12. **Agentic fund exfiltration to mule accounts** - once an account is
    taken over, an agent automates the transfer sequence to mule accounts,
    sometimes layering across several to obscure origin - the pattern
    behind incidents like the 2024 Flutterwave Nigeria breach (~$7M routed
    through multiple institutions). [arXiv 2606.17555]
13. **Coordinated mule-account networks / structuring** - fraud rings using
    AI to coordinate synchronized account openings and card testing across
    many mule identities at once, not just one account at a time.
    [Group-IB 2026]

## E. Scale economics (why this is a step-change, not just "more fraud")

14. **Fraud-as-a-service** - GenAI lowers the skill floor enough that
    autonomous fraud agents and attack kits are rented by the month,
    putting professional-grade multi-vector attacks into amateur hands.
    [bravenewcoin.com Fraud Trends 2026]
15. **Adaptive, institution-specific attacks** - the same GenAI capability
    that generates an attack can also read a target institution's
    observable defenses and iterate against them - this is the premise the
    whole competition is built on, and it's the direct real-world analogue
    of this project's own Red Team evolution loop.

## Coverage matrix: what Generate/Defend actually simulates today

| # | Attack vector | Modality | Simulated by red_team.py? |
|---|---|---|---|
| 1 | Synthetic identity | Identity/PII + transaction | Partial - "BRAND_NEW" customer + thin-file pattern, no real identity-graph modeling |
| 2 | Deepfake liveness bypass | Video/biometric | **Not simulated** - no biometric data in this system |
| 3 | Document deepfakes | Image/document | **Not simulated** |
| 4 | Deepfake voice vs call center | Audio | **Not simulated** |
| 5 | Real-time deepfake video auth | Video | **Not simulated** |
| 6 | Agentic account takeover | Transaction + device | Yes - `device` gene (new device, sustained use) |
| 7 | AI-generated phishing/BEC | Text/email | **Not simulated** - out of scope (no email/comms channel modeled) |
| 8 | Deepfake CEO fraud (wire auth) | Audio/video + transaction | Partial - `amount` abrupt_spike approximates the resulting transaction, not the social-engineering trigger |
| 9 | Multi-vector campaigns | Cross-channel | **Not simulated** - single payment channel only |
| 10 | Pig-butchering via LLM chat | Text/relationship | **Not simulated** |
| 11 | Automated card testing | Transaction | **Not simulated until this pass** - see below |
| 12 | Agentic fund exfiltration | Transaction + account graph | Yes - `coordination` gene (multi-account spread) |
| 13 | Coordinated mule networks | Cross-customer graph | Partial - `coordination` only spans one customer's own accounts, not a ring of different customer_ids |
| 14 | Fraud-as-a-service scale | N/A (economic pattern) | N/A - reflected in population size, not a single strategy |
| 15 | Adaptive institution-specific attack | N/A (meta-pattern) | Yes - this is the evolutionary loop itself |

**Honest read of this table**: this system covers transaction-pattern fraud
well (account takeover, fund movement, structuring, and now card testing)
but has zero coverage of the identity/biometric/communications attack
surface (rows 2-5, 7, 9, 10) that a large share of current GenAI fraud
research is actually about. That's a real scope boundary, not an oversight
being hidden - a full submission against this brief would need a second,
entirely different Generate/Defend pipeline (image/audio/text-based) to
cover those rows, which is out of reach in this session.

## What this pass adds to Generate, directly from row 11

`red_team.py`'s `amount_pattern` previously had `gradual_drift` and
`abrupt_spike` - both change the *size* of transactions, neither one models
*probing*: several small, easily-dismissed transactions before a real
attempt, which is specifically how card testing works and is structurally
different from either existing pattern. Added as `card_testing`: N small
probe amounts ($0.50-$5, matching real-world reporting of "small, unnoticed
purchases") followed by one larger transaction at the same merchant
category, all in a short window. See red_team.py's GenomeCodec._amount.
