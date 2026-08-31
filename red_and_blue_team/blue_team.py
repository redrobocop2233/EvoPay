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


# ---------------------------------------------------------------------------
# Trainable detector — wraps a RandomForestClassifier that can be retrained
# every generation to close the loop. Before the first training call it
# delegates to HeuristicDetector so the first generation still has a scorer.
# ---------------------------------------------------------------------------

import copy
import numpy as np


class TrainableDetector(DetectorInterface):
    """ML-backed detector that retrains on accumulated labeled campaigns."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self._model = None
        self._heuristic = HeuristicDetector(threshold=threshold)
        self._training_X = []
        self._training_y = []

    # --- DetectorInterface ---------------------------------------------------

    def evaluate(self, transactions, ecosystem):
        if not transactions:
            return DetectionResult(detected=False, risk_score=0.0, reason_codes=[])

        # Before first training, delegate to the heuristic
        if self._model is None:
            return self._heuristic.evaluate(transactions, ecosystem)

        features = self._featurize(transactions, ecosystem)
        proba = float(self._model.predict_proba([features])[:, 1][0])
        detected = proba >= self.threshold

        # Still collect reason codes from the heuristic for explainability
        heuristic_result = self._heuristic.evaluate(transactions, ecosystem)
        return DetectionResult(
            detected=detected,
            risk_score=round(proba, 3),
            reason_codes=heuristic_result.reason_codes,
        )

    def score_campaign(self, transactions, ecosystem):
        """Return probability [0,1] that this campaign is flagged fraudulent."""
        result = self.evaluate(transactions, ecosystem)
        return result.risk_score

    # --- Retraining ----------------------------------------------------------

    def add_training_data(self, transactions, ecosystem, label):
        """Accumulate labeled campaign data for training."""
        features = self._featurize(transactions, ecosystem)
        self._training_X.append(features)
        self._training_y.append(label)

    def update(self, campaign_records, campaign_txns_map, ecosystem):
        """Retrain the model on all accumulated data.

        Args:
            campaign_records: list of MemoryRecord from this generation
            campaign_txns_map: dict mapping campaign_id -> list of transactions
            ecosystem: the PaymentEcosystem for featurizing
        """
        # Add this generation's labeled campaigns
        for record in campaign_records:
            txns = campaign_txns_map.get(record.campaign_id)
            if txns:
                features = self._featurize(txns, ecosystem)
                self._training_X.append(features)
                self._training_y.append(1)  # fraud campaigns

        # Need at least some samples of each class
        if sum(self._training_y) < 2 or len(self._training_y) - sum(self._training_y) < 2:
            return  # not enough data yet

        from sklearn.ensemble import RandomForestClassifier
        X = np.array(self._training_X)
        y = np.array(self._training_y)
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def add_legitimate_baseline(self, ecosystem, rng, n=50):
        """Add legitimate transaction samples as class-0 training data."""
        if not ecosystem.transactions:
            return
        sample = rng.sample(
            ecosystem.transactions,
            min(n, len(ecosystem.transactions)),
        )
        # Group by customer_id for featurization
        from collections import defaultdict
        by_customer = defaultdict(list)
        for txn in sample:
            by_customer[txn.customer_id].append(txn)
        for cust_txns in by_customer.values():
            features = self._featurize(cust_txns, ecosystem)
            self._training_X.append(features)
            self._training_y.append(0)

    def frozen_copy(self):
        """Return a deep copy that will never be updated — for static baseline."""
        return copy.deepcopy(self)

    # --- Feature extraction --------------------------------------------------

    def _featurize(self, transactions, ecosystem):
        """Extract numeric features from a campaign's transactions.

        Mirrors the signals the HeuristicDetector checks but as numeric
        features suitable for a classifier.
        """
        if not transactions:
            return np.zeros(10)

        t0 = transactions[0]

        # Customer lookup
        customer = None
        for c in ecosystem.customers:
            if c.customer_id == t0.customer_id:
                customer = c
                break

        # Account lookup for device info
        known_devices = set()
        for a in ecosystem.accounts:
            if a.account_id == t0.account_id:
                known_devices = set(a.device_ids)
                break

        amounts = [t.amount for t in transactions]
        avg_amount = np.mean(amounts)
        max_amount = max(amounts)
        num_txns = len(transactions)

        # 1. Amount deviation from customer baseline
        if customer and customer.amount_std > 0:
            amount_z = abs(avg_amount - customer.avg_amount) / customer.amount_std
        else:
            amount_z = 0.0

        # 2. Max amount deviation
        if customer and customer.amount_std > 0:
            max_amount_z = abs(max_amount - customer.avg_amount) / customer.amount_std
        else:
            max_amount_z = 0.0

        # 3. Off-hours ratio
        off_hours_count = 0
        for t in transactions:
            hour = int(t.timestamp[11:13])
            if customer:
                start, end = customer.active_hours
                if hour < start or hour > end:
                    off_hours_count += 1
            else:
                if hour < 6 or hour > 22:
                    off_hours_count += 1
        off_hours_ratio = off_hours_count / num_txns

        # 4. Unknown device ratio
        unknown_device_count = sum(1 for t in transactions if t.device_id not in known_devices)
        unknown_device_ratio = unknown_device_count / num_txns

        # 5. Unusual location ratio
        unusual_location_count = 0
        for t in transactions:
            if customer:
                weight = customer.location_weights.get(t.city, 0.0)
                if weight < 0.3:
                    unusual_location_count += 1
            else:
                unusual_location_count += 1
        unusual_location_ratio = unusual_location_count / num_txns

        # 6. Velocity (txns per minute span)
        timestamps = sorted(datetime.fromisoformat(t.timestamp) for t in transactions)
        if len(timestamps) >= 2:
            span_minutes = max(1, (timestamps[-1] - timestamps[0]).total_seconds() / 60)
            velocity = num_txns / span_minutes
        else:
            velocity = 0.0

        # 7. Transaction count
        txn_count = float(num_txns)

        # 8. Amount variance (high variance = card testing pattern)
        amount_std = float(np.std(amounts)) if len(amounts) > 1 else 0.0

        # 9. Small amount count (card testing signal)
        small_amount_count = sum(1 for a in amounts if a < 10.0)
        small_amount_ratio = small_amount_count / num_txns

        # 10. Number of distinct merchants
        distinct_merchants = len(set(t.merchant_category for t in transactions))
        merchant_diversity = distinct_merchants / max(1, num_txns)

        return np.array([
            min(amount_z, 10.0),        # 0: amount_z_score
            min(max_amount_z, 15.0),    # 1: max_amount_z_score
            off_hours_ratio,            # 2: off_hours_ratio
            unknown_device_ratio,       # 3: unknown_device_ratio
            unusual_location_ratio,     # 4: unusual_location_ratio
            min(velocity, 10.0),        # 5: velocity
            min(txn_count, 10.0),       # 6: txn_count
            min(amount_std, 5000.0),    # 7: amount_std
            small_amount_ratio,         # 8: small_amount_ratio
            merchant_diversity,         # 9: merchant_diversity
        ])

