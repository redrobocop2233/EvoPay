"""
GenAI Attack Autopsy (v2 per the design doc).

Runs strictly AFTER Blue's detection decision - never influences risk_score
or decision, only explains it and suggests a mutation direction for the
evolutionary engine's next generation. The mutation direction is advisory:
EvolutionEngine's actual mutation is still the deterministic gaussian
perturbation in red_team.py. This is interpretation, not control.
"""

from .client import GenAIClient
from .prompts import AUTOPSY_SYSTEM, autopsy_user_prompt
from .schemas import AttackAutopsy


class AttackAnalyst:
    def __init__(self, client: GenAIClient | None = None):
        self.client = client or GenAIClient()

    def analyze(self, strategy_id: str, strategy_summary: dict, detection_result: dict) -> AttackAutopsy:
        user = autopsy_user_prompt(strategy_summary, detection_result)
        autopsy = self.client.complete_json_single(AUTOPSY_SYSTEM, user, AttackAutopsy)
        # the model doesn't reliably echo strategy_id/blue_risk_score/detected
        # back verbatim even when told to - pin them from ground truth rather
        # than trust the LLM's copy of numbers it was already given
        autopsy.strategy_id = strategy_id
        autopsy.blue_risk_score = detection_result.get("risk_score", autopsy.blue_risk_score)
        autopsy.detected = detection_result.get("detected", autopsy.detected)
        return autopsy
