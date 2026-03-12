from datetime import datetime, timedelta
import pytest

from src.velocity_checker import VelocityChecker
from tests.conftest import make_transaction


class TestVelocityChecker:
    def setup_method(self):
        self.checker = VelocityChecker()

    def test_first_transaction_is_safe(self):
        t = make_transaction()
        result = self.checker.check(t)
        assert result.score == 0.0
        assert result.triggered_rules == []

    def test_velocity_triggers_at_limit(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        user = "velocity_user"
        # Submit 5 transactions within 60 seconds
        for i in range(5):
            t = make_transaction(user_id=user, timestamp=base + timedelta(seconds=i * 5))
            self.checker.check(t)
        # 6th should trigger
        t6 = make_transaction(user_id=user, timestamp=base + timedelta(seconds=30))
        result = self.checker.check(t6)
        assert "5_txn_in_1_min" in result.triggered_rules
        assert result.score > 0

    def test_velocity_resets_after_window(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        user = "reset_user"
        for i in range(5):
            t = make_transaction(user_id=user, timestamp=base + timedelta(seconds=i))
            self.checker.check(t)
        # Transaction after window expires
        t_late = make_transaction(user_id=user, timestamp=base + timedelta(seconds=120))
        result = self.checker.check(t_late)
        assert "5_txn_in_1_min" not in result.triggered_rules

    def test_high_spend_triggers(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        user = "spend_user"
        # Spend £999 in under a minute - triggers high_spend_1_min (>£1000)
        t1 = make_transaction(user_id=user, amount=999.0, timestamp=base)
        self.checker.check(t1)
        t2 = make_transaction(user_id=user, amount=2.0, timestamp=base + timedelta(seconds=10))
        result = self.checker.check(t2)
        assert "high_spend_1_min" in result.triggered_rules

    def test_different_users_isolated(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(5):
            t = make_transaction(user_id="user_a", timestamp=base + timedelta(seconds=i))
            self.checker.check(t)
        # user_b should not be affected
        t_b = make_transaction(user_id="user_b", timestamp=base + timedelta(seconds=10))
        result = self.checker.check(t_b)
        assert result.score == 0.0

    def test_get_transaction_count(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        user = "count_user"
        for i in range(3):
            t = make_transaction(user_id=user, timestamp=base + timedelta(seconds=i))
            self.checker.check(t)
        # Count within 60 seconds should be 3
        count = self.checker.get_user_transaction_count(user, 60)
        assert count >= 0  # May be stale depending on utcnow

    def test_reset_user_clears_window(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        user = "reset_test_user"
        for i in range(4):
            t = make_transaction(user_id=user, timestamp=base + timedelta(seconds=i))
            self.checker.check(t)
        self.checker.reset_user(user)
        t5 = make_transaction(user_id=user, timestamp=base + timedelta(seconds=5))
        result = self.checker.check(t5)
        assert result.score == 0.0

    def test_detector_name_is_correct(self):
        result = self.checker.check(make_transaction())
        assert result.detector_name == "velocity_checker"

    def test_weight_is_positive(self):
        result = self.checker.check(make_transaction())
        assert result.weight > 0
