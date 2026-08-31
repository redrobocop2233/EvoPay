"""
Client for the real Blue Team API (Blue-Team-mastercard-main), implementing
the same DetectorInterface as blue_team.HeuristicDetector. RedTeamController
doesn't need to know which one it's talking to - swapping the local heuristic
for the real system is a one-line change (see red_team.py's __main__ block).

Blue's /evaluate endpoint scores one transaction at a time and expects a
customer_history array so it can build a behavioral profile and temporal
features for customers it has never seen (which is every Red Team customer,
since Red and Blue run separate synthetic worlds). A campaign becomes one
/evaluate call per transaction; the campaign-level DetectionResult is the
max risk across those calls (any one caught transaction means the campaign
was caught) with the union of reason codes.
"""

import time
from dataclasses import asdict

import requests

from .blue_team import DetectorInterface, DetectionResult

# fields on our FraudTransaction/Transaction that aren't part of Blue's
# TransactionContext.transaction payload - Blue only wants the raw
# transaction fields, not our internal bookkeeping
_INTERNAL_FIELDS = {"transaction_id", "campaign_id", "strategy_id", "account_id",
                     "label", "is_fraud", "attack_family", "customer_id"}


class BlueTeamClient(DetectorInterface):
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=15, fallback=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fallback = fallback  # e.g. HeuristicDetector(), used if the API is unreachable
        self._warned_fallback = False

    def evaluate(self, transactions, ecosystem):
        if not transactions:
            return DetectionResult(detected=False, risk_score=0.0, reason_codes=[])

        customer_id = transactions[0].customer_id
        history = self._history_payload(customer_id, ecosystem)

        risk_scores = []
        detected_any = False
        reason_codes = set()

        for txn in transactions:
            try:
                result = self._evaluate_one(txn, history)
            except (requests.RequestException, ValueError) as e:
                return self._fallback_result(transactions, ecosystem, reason="api_error", detail=str(e))

            risk_scores.append(result["risk_score"])
            detected_any = detected_any or result["detected"]
            reason_codes.update(result.get("reason_codes", []))

        return DetectionResult(
            detected=detected_any,
            risk_score=round(max(risk_scores), 3) if risk_scores else 0.0,
            reason_codes=sorted(reason_codes),
        )

    def _evaluate_one(self, txn, history):
        payload = {
            "campaign_id": getattr(txn, "campaign_id", "legit"),
            "strategy_id": getattr(txn, "strategy_id", None),
            "customer_id": txn.customer_id,
            "transaction": self._transaction_payload(txn),
            "customer_history": history,
        }
        response = requests.post(f"{self.base_url}/evaluate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _transaction_payload(self, txn):
        row = asdict(txn)
        return {k: v for k, v in row.items() if k not in _INTERNAL_FIELDS}

    def _history_payload(self, customer_id, ecosystem, max_rows=200):
        # legit history only - a real fraud detector builds its baseline
        # from legitimate behavior, not from other attacks
        rows = [asdict(t) for t in ecosystem.transactions if t.customer_id == customer_id]
        rows = rows[-max_rows:]
        for row in rows:
            row.pop("account_id", None)
        return rows

    def _fallback_result(self, transactions, ecosystem, reason, detail):
        if self.fallback is None:
            raise RuntimeError(f"Blue Team API unreachable ({reason}: {detail}) and no fallback detector set")
        if not self._warned_fallback:
            print(f"[BlueTeamClient] API unreachable ({reason}: {detail}) - "
                  f"falling back to local heuristic detector for this run")
            self._warned_fallback = True
        return self.fallback.evaluate(transactions, ecosystem)


def wait_for_api(base_url="http://127.0.0.1:8000", timeout=15):
    """Poll /health until the Blue Team API is up, or give up after timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Blue Team API at {base_url} did not become healthy within {timeout}s")
