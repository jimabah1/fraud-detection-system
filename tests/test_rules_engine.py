import pytest
from datetime import datetime

from src.rules_engine import RulesEngine
from src.models import TransactionType
from tests.conftest import make_transaction


class TestRulesEngine:
    def setup_method(self):
        self.engine = RulesEngine()

    def test_clean_transaction_no_triggers(self):
        t = make_transaction()
        result = self.engine.check(t)
        assert result.score == 0.0
        assert result.triggered_rules == []

    def test_blocked_user_max_score(self):
        self.engine.block_user("bad_user")
        t = make_transaction(user_id="bad_user")
        result = self.engine.check(t)
        assert result.score == 100.0
        assert "blocked_user" in result.triggered_rules

    def test_blocked_merchant_max_score(self):
        self.engine.block_merchant("bad_merchant")
        t = make_transaction(merchant_id="bad_merchant")
        result = self.engine.check(t)
        assert result.score == 100.0
        assert "blocked_merchant" in result.triggered_rules

    def test_unblock_user(self):
        self.engine.block_user("temp_blocked")
        self.engine.unblock_user("temp_blocked")
        t = make_transaction(user_id="temp_blocked")
        result = self.engine.check(t)
        assert result.score == 0.0

    def test_new_device_flagged(self):
        user = "device_user"
        # First transaction with device_1 (not flagged - no history)
        t1 = make_transaction(user_id=user, device_id="device_1")
        self.engine.check(t1)
        # Second transaction with new device
        t2 = make_transaction(user_id=user, device_id="device_2")
        result = self.engine.check(t2)
        assert "new_device" in result.triggered_rules

    def test_same_device_not_flagged(self):
        user = "same_device_user"
        t1 = make_transaction(user_id=user, device_id="device_a")
        self.engine.check(t1)
        t2 = make_transaction(user_id=user, device_id="device_a")
        result = self.engine.check(t2)
        assert "new_device" not in result.triggered_rules

    def test_new_merchant_category_flagged(self):
        user = "cat_user"
        t1 = make_transaction(user_id=user, merchant_category="retail")
        self.engine.check(t1)
        t2 = make_transaction(user_id=user, merchant_category="gambling")
        result = self.engine.check(t2)
        assert "new_merchant_category" in result.triggered_rules

    def test_high_risk_merchant_category(self):
        t = make_transaction(merchant_category="gambling")
        result = self.engine.check(t)
        assert "high_risk_merchant_category" in result.triggered_rules
        assert result.score >= 40

    def test_after_hours_transaction(self):
        # 03:00 UTC - after hours
        t = make_transaction(timestamp=datetime(2024, 1, 1, 3, 0, 0))
        result = self.engine.check(t)
        assert "after_hours_transaction" in result.triggered_rules

    def test_daytime_transaction_not_after_hours(self):
        t = make_transaction(timestamp=datetime(2024, 1, 1, 12, 0, 0))
        result = self.engine.check(t)
        assert "after_hours_transaction" not in result.triggered_rules

    def test_large_international_transfer(self):
        t = make_transaction(
            amount=3000.0,
            transaction_type=TransactionType.TRANSFER,
            is_international=True,
        )
        result = self.engine.check(t)
        assert "large_international_transfer" in result.triggered_rules
        assert result.score >= 55

    def test_large_withdrawal_flagged(self):
        t = make_transaction(
            amount=600.0,
            transaction_type=TransactionType.WITHDRAWAL,
        )
        result = self.engine.check(t)
        assert "large_cash_withdrawal" in result.triggered_rules

    def test_multiple_signals_compound(self):
        user = "compound_user"
        # Establish device and category
        self.engine.check(make_transaction(user_id=user, device_id="d1", merchant_category="retail"))
        # High-risk: new device + gambling + after hours
        t = make_transaction(
            user_id=user,
            device_id="d2",
            merchant_category="gambling",
            timestamp=datetime(2024, 1, 1, 2, 0, 0),
        )
        result = self.engine.check(t)
        assert result.score > 40
        assert len(result.triggered_rules) >= 3

    def test_get_user_profile(self):
        user = "profile_user"
        self.engine.check(make_transaction(user_id=user, merchant_category="retail", device_id="d1"))
        profile = self.engine.get_user_profile(user)
        assert "retail" in profile["merchant_categories"]
        assert "d1" in profile["known_devices"]
        assert profile["is_blocked"] is False

    def test_reset_user(self):
        user = "rules_reset"
        self.engine.check(make_transaction(user_id=user, merchant_category="retail", device_id="d1"))
        self.engine.reset_user(user)
        profile = self.engine.get_user_profile(user)
        assert profile["merchant_categories"] == []
        assert profile["known_devices"] == []

    def test_detector_name(self):
        result = self.engine.check(make_transaction())
        assert result.detector_name == "rules_engine"

    def test_weight_is_highest(self):
        result = self.engine.check(make_transaction())
        assert result.weight == 2.0
