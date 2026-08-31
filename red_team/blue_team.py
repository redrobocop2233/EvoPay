"""
Blue-team detector, v0.3.

Gives the red team an actual opponent to query instead of the anomaly_score
proxy from v0.2. HeuristicDetector is a transparent, rule-based baseline -
not a trained model - that scores a campaign's transactions against the
transacting customer's known behavior: typical amount, active hours,
location distribution, known devices, and transaction velocity.

A learned detector (the tabular/temporal/graph ensemble from the original
project concept) can implement the same DetectorInterface later without
requiring any change on the red-team side - only evaluate() needs to swap.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class DetectionResult:
    detected: bool
    risk_score: float
    reason_codes: list


class DetectorInterface:
    def evaluate(self, transactions, ecosystem):
        raise NotImplementedError


class DummyDetector(DetectorInterface):
    """Never flags anything - a no-op baseline, useful only for wiring
    the interface together before a real detector exists."""

    def evaluate(self, transactions, ecosystem):
        return DetectionResult(detected=False, risk_score=0.0, reason_codes=[])


class HeuristicDetector(DetectorInterface):
    """Rule-based baseline: one signal per behavioral dimension, weighted
    and combined into a single risk score."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def evaluate(self, transactions, ecosystem):
        if not transactions:
            return DetectionResult(detected=False, risk_score=0.0, reason_codes=[])

        customer = self._customer_for(transactions[0].customer_id, ecosystem)
        account = self._account_for(transactions[0].account_id, ecosystem)
        known_devices = set(account.device_ids)

        per_txn_scores = []
        reason_codes = set()
        for t in transactions:
            score, reasons = self._score_transaction(t, customer, known_devices)
            per_txn_scores.append(score)
            reason_codes.update(reasons)

        velocity_score, velocity_reason = self._score_velocity(transactions)
        if velocity_reason:
            reason_codes.add(velocity_reason)

        risk_score = min(1.0, (sum(per_txn_scores) / len(per_txn_scores)) + velocity_score)
        detected = risk_score >= self.threshold
        return DetectionResult(detected=detected, risk_score=round(risk_score, 3),
                                reason_codes=sorted(reason_codes))

    def _customer_for(self, customer_id, ecosystem):
        return next(c for c in ecosystem.customers if c.customer_id == customer_id)

    def _account_for(self, account_id, ecosystem):
        return next(a for a in ecosystem.accounts if a.account_id == account_id)

    def _score_transaction(self, t, customer, known_devices):
        score = 0.0
        reasons = []

        if customer.amount_std > 0:
            z = abs(t.amount - customer.avg_amount) / customer.amount_std
            amount_signal = min(1.0, z / 6.0)
        else:
            amount_signal = 0.0
        if amount_signal > 0.15:
            reasons.append("amount_deviation")
        score += 0.35 * amount_signal

        hour = int(t.timestamp[11:13])
        start, end = customer.active_hours
        hour_signal = 0.0 if start <= hour <= end else 1.0
        if hour_signal:
            reasons.append("off_hours")
        score += 0.20 * hour_signal

        city_weight = customer.location_weights.get(t.city, 0.0)
        city_signal = 1.0 - city_weight
        if city_signal > 0.7:
            reasons.append("unusual_location")
        score += 0.20 * city_signal

        device_signal = 0.0 if t.device_id in known_devices else 1.0
        if device_signal:
            reasons.append("unknown_device")
        score += 0.25 * device_signal

        return min(1.0, score), reasons

    def _score_velocity(self, transactions):
        if len(transactions) < 3:
            return 0.0, None
        timestamps = sorted(datetime.fromisoformat(t.timestamp) for t in transactions)
        span_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60
        if span_minutes < 30:
            return 0.25, "velocity_burst"
        return 0.0, None
