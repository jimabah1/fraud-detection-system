import threading
from collections import deque
from datetime import datetime, timedelta

from .models import Transaction, DetectorResult


class VelocityChecker:
    """
    Detects unusually high transaction frequency using a sliding window.

    Uses collections.deque for O(1) amortized appends and pops, making
    this safe for high-throughput real-time processing.
    """

    DEFAULT_RULES = [
        {"name": "5_txn_in_1_min",   "window_seconds": 60,   "max_count": 5,   "score": 40},
        {"name": "10_txn_in_5_min",  "window_seconds": 300,  "max_count": 10,  "score": 60},
        {"name": "20_txn_in_1_hour", "window_seconds": 3600, "max_count": 20,  "score": 80},
        {"name": "high_spend_1_min", "window_seconds": 60,   "max_amount": 1000.0, "score": 50},
        {"name": "high_spend_1_hour","window_seconds": 3600, "max_amount": 5000.0, "score": 70},
    ]

    def __init__(self, rules: list[dict] | None = None):
        self._rules = rules or self.DEFAULT_RULES
        # user_id -> deque of (timestamp, amount)
        self._windows: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _get_window(self, user_id: str) -> deque:
        if user_id not in self._windows:
            self._windows[user_id] = deque()
        return self._windows[user_id]

    def _evict_old_entries(self, window: deque, cutoff: datetime) -> None:
        while window and window[0][0] < cutoff:
            window.popleft()

    def check(self, transaction: Transaction) -> DetectorResult:
        triggered = []
        max_score = 0.0

        with self._lock:
            window = self._get_window(transaction.user_id)
            now = transaction.timestamp

            for rule in self._rules:
                cutoff = now - timedelta(seconds=rule["window_seconds"])
                self._evict_old_entries(window, cutoff)

                if "max_count" in rule:
                    count = len(window)
                    if count >= rule["max_count"]:
                        triggered.append(rule["name"])
                        max_score = max(max_score, rule["score"])

                elif "max_amount" in rule:
                    total = sum(amt for _, amt in window)
                    if total + transaction.amount > rule["max_amount"]:
                        triggered.append(rule["name"])
                        max_score = max(max_score, rule["score"])

            window.append((now, transaction.amount))

        return DetectorResult(
            detector_name="velocity_checker",
            score=max_score,
            weight=1.5,
            triggered_rules=triggered,
            metadata={"window_size": len(self._get_window(transaction.user_id))},
        )

    def get_user_transaction_count(self, user_id: str, window_seconds: int) -> int:
        with self._lock:
            window = self._windows.get(user_id)
            if not window:
                return 0
            cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
            return sum(1 for ts, _ in window if ts >= cutoff)

    def reset_user(self, user_id: str) -> None:
        with self._lock:
            self._windows.pop(user_id, None)
