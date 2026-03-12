import math
import threading

from .models import Transaction, DetectorResult


class WelfordStats:
    """
    Online mean and variance using Welford's algorithm.
    O(1) space and O(1) per update - no need to store transaction history.

    Reference: Welford, B. P. (1962). "Note on a method for calculating
    corrected sums of squares and products."
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._M2 = 0.0  # sum of squared deviations

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self._M2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)

    def z_score(self, value: float) -> float:
        if self.std_dev == 0:
            return 0.0
        return (value - self.mean) / self.std_dev


class AmountAnalyzer:
    """
    Detects statistically anomalous transaction amounts using per-user
    rolling statistics (Welford's online algorithm).

    Also applies rule-based heuristics:
    - Round number bias (fraudsters often test with round amounts)
    - Micro-transactions (card testing pattern: many small amounts)
    - Just-below threshold amounts (e.g. £999 to avoid £1000 limit)
    """

    Z_SCORE_REVIEW = 2.5
    Z_SCORE_SUSPICIOUS = 3.5
    Z_SCORE_BLOCK = 5.0

    ABSOLUTE_HIGH_AMOUNT = 10_000.0
    MICRO_TRANSACTION_MAX = 1.0
    JUST_BELOW_THRESHOLDS = [500.0, 1000.0, 2000.0, 5000.0, 10000.0]
    JUST_BELOW_MARGIN = 0.05  # within 5% below threshold

    MIN_SAMPLES_FOR_Z_SCORE = 5

    def __init__(self):
        # user_id -> WelfordStats
        self._user_stats: dict[str, WelfordStats] = {}
        self._lock = threading.Lock()

    def _get_stats(self, user_id: str) -> WelfordStats:
        if user_id not in self._user_stats:
            self._user_stats[user_id] = WelfordStats()
        return self._user_stats[user_id]

    def _is_round_number(self, amount: float) -> bool:
        return amount >= 10 and amount % 10 == 0

    def _is_just_below_threshold(self, amount: float) -> bool:
        for threshold in self.JUST_BELOW_THRESHOLDS:
            lower = threshold * (1 - self.JUST_BELOW_MARGIN)
            if lower <= amount < threshold:
                return True
        return False

    def check(self, transaction: Transaction) -> DetectorResult:
        triggered = []
        score = 0.0
        metadata = {}

        amount = transaction.amount

        with self._lock:
            stats = self._get_stats(transaction.user_id)

            if stats.count >= self.MIN_SAMPLES_FOR_Z_SCORE:
                z = stats.z_score(amount)
                metadata["z_score"] = round(z, 2)
                metadata["user_mean"] = round(stats.mean, 2)
                metadata["user_std_dev"] = round(stats.std_dev, 2)

                if z >= self.Z_SCORE_BLOCK:
                    triggered.append("extreme_amount_outlier")
                    score = max(score, 85.0)
                elif z >= self.Z_SCORE_SUSPICIOUS:
                    triggered.append("high_amount_outlier")
                    score = max(score, 65.0)
                elif z >= self.Z_SCORE_REVIEW:
                    triggered.append("amount_outlier")
                    score = max(score, 40.0)
            else:
                metadata["samples"] = stats.count
                metadata["note"] = "insufficient_history_for_z_score"

            stats.update(amount)

        if amount >= self.ABSOLUTE_HIGH_AMOUNT:
            triggered.append("absolute_high_amount")
            score = max(score, 55.0)

        if amount <= self.MICRO_TRANSACTION_MAX:
            triggered.append("micro_transaction")
            score = max(score, 30.0)

        if self._is_just_below_threshold(amount):
            triggered.append("just_below_threshold")
            score = max(score, 35.0)

        if self._is_round_number(amount):
            triggered.append("round_number_bias")
            score = max(score, score + 5.0)

        metadata["amount"] = amount

        return DetectorResult(
            detector_name="amount_analyzer",
            score=min(score, 100.0),
            weight=1.0,
            triggered_rules=triggered,
            metadata=metadata,
        )

    def get_user_stats(self, user_id: str) -> dict:
        with self._lock:
            stats = self._user_stats.get(user_id)
            if not stats:
                return {"count": 0}
            return {
                "count": stats.count,
                "mean": round(stats.mean, 2),
                "std_dev": round(stats.std_dev, 2),
            }

    def reset_user(self, user_id: str) -> None:
        with self._lock:
            self._user_stats.pop(user_id, None)
