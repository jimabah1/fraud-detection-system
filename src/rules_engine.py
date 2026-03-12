import threading
from datetime import datetime

from .models import Transaction, TransactionType, DetectorResult


class RulesEngine:
    """
    Rule-based fraud detection for deterministic, explainable signals.

    These rules catch known fraud patterns that ML models might miss
    during cold-start, and provide human-readable audit trails.
    """

    # Merchant categories associated with high fraud rates
    HIGH_RISK_CATEGORIES = {
        "gambling", "crypto", "adult_content", "wire_transfer",
        "money_order", "prepaid_cards", "pawn_shop",
    }

    # After-hours window (UTC): transactions between 01:00 - 05:00
    AFTER_HOURS_START = 1
    AFTER_HOURS_END = 5

    def __init__(self):
        # user_id -> set of merchant categories seen
        self._user_merchant_categories: dict[str, set] = {}
        # user_id -> set of device_ids seen
        self._user_devices: dict[str, set] = {}
        # Blocked user IDs
        self._blocked_users: set[str] = set()
        # Blocked merchant IDs
        self._blocked_merchants: set[str] = set()
        self._lock = threading.Lock()

    def block_user(self, user_id: str) -> None:
        with self._lock:
            self._blocked_users.add(user_id)

    def block_merchant(self, merchant_id: str) -> None:
        with self._lock:
            self._blocked_merchants.add(merchant_id)

    def unblock_user(self, user_id: str) -> None:
        with self._lock:
            self._blocked_users.discard(user_id)

    def check(self, transaction: Transaction) -> DetectorResult:
        triggered = []
        score = 0.0

        with self._lock:
            # Hard blocks - instant maximum score
            if transaction.user_id in self._blocked_users:
                triggered.append("blocked_user")
                score = 100.0

            if transaction.merchant_id and transaction.merchant_id in self._blocked_merchants:
                triggered.append("blocked_merchant")
                score = 100.0

            if score == 100.0:
                return DetectorResult(
                    detector_name="rules_engine",
                    score=100.0,
                    weight=2.0,
                    triggered_rules=triggered,
                )

            # New device for this user
            devices = self._user_devices.setdefault(transaction.user_id, set())
            if transaction.device_id and transaction.device_id not in devices:
                if len(devices) > 0:
                    triggered.append("new_device")
                    score = max(score, 25.0)
                devices.add(transaction.device_id)

            # New merchant category for this user
            categories = self._user_merchant_categories.setdefault(transaction.user_id, set())
            if transaction.merchant_category not in categories:
                if len(categories) > 0:
                    triggered.append("new_merchant_category")
                    score = max(score, 15.0)
                categories.add(transaction.merchant_category)

        # High-risk merchant category
        if transaction.merchant_category.lower() in self.HIGH_RISK_CATEGORIES:
            triggered.append("high_risk_merchant_category")
            score = max(score, 40.0)

        # After-hours transaction
        hour = transaction.timestamp.hour
        if self.AFTER_HOURS_START <= hour < self.AFTER_HOURS_END:
            triggered.append("after_hours_transaction")
            score = max(score, 20.0)

        # Large international transfer
        if (
            transaction.is_international
            and transaction.transaction_type == TransactionType.TRANSFER
            and transaction.amount >= 2000.0
        ):
            triggered.append("large_international_transfer")
            score = max(score, 55.0)

        # Withdrawal exceeds typical threshold
        if (
            transaction.transaction_type == TransactionType.WITHDRAWAL
            and transaction.amount >= 500.0
        ):
            triggered.append("large_cash_withdrawal")
            score = max(score, 30.0)

        # Multiple high-risk signals compound
        if len(triggered) >= 3:
            score = min(score + 15.0, 100.0)

        return DetectorResult(
            detector_name="rules_engine",
            score=min(score, 100.0),
            weight=2.0,
            triggered_rules=triggered,
        )

    def get_user_profile(self, user_id: str) -> dict:
        with self._lock:
            return {
                "merchant_categories": list(self._user_merchant_categories.get(user_id, set())),
                "known_devices": list(self._user_devices.get(user_id, set())),
                "is_blocked": user_id in self._blocked_users,
            }

    def reset_user(self, user_id: str) -> None:
        with self._lock:
            self._user_merchant_categories.pop(user_id, None)
            self._user_devices.pop(user_id, None)
            self._blocked_users.discard(user_id)
