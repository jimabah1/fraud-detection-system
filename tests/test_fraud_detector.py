from datetime import datetime, timedelta
import pytest

from src.fraud_detector import FraudDetector
from src.models import RiskLevel, TransactionType
from tests.conftest import make_transaction, make_location


class TestFraudDetector:
    def setup_method(self):
        self.detector = FraudDetector()

    def test_clean_transaction_is_safe(self):
        t = make_transaction(amount=25.0)
        result = self.detector.evaluate(t)
        assert result.risk_level == RiskLevel.SAFE
        assert result.final_score <= 30

    def test_blocked_user_is_blocked(self):
        self.detector.block_user("blocked_u")
        t = make_transaction(user_id="blocked_u")
        result = self.detector.evaluate(t)
        assert result.risk_level == RiskLevel.BLOCK
        assert result.recommended_action == "DECLINE"

    def test_blocked_merchant_is_blocked(self):
        self.detector.block_merchant("blocked_m")
        t = make_transaction(merchant_id="blocked_m")
        result = self.detector.evaluate(t)
        assert result.risk_level == RiskLevel.BLOCK

    def test_unblock_user(self):
        self.detector.block_user("temp_u")
        self.detector.unblock_user("temp_u")
        t = make_transaction(user_id="temp_u")
        result = self.detector.evaluate(t)
        assert result.risk_level != RiskLevel.BLOCK

    def test_all_five_detectors_present(self):
        t = make_transaction()
        result = self.detector.evaluate(t)
        names = {d.detector_name for d in result.detector_results}
        assert "rules_engine" in names
        assert "velocity_checker" in names
        assert "geo_analyzer" in names
        assert "amount_analyzer" in names
        assert "ml_scorer" in names

    def test_impossible_travel_raises_score(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        london = make_location(51.5, -0.1, "GB")
        ny = make_location(40.7, -74.0, "US")

        t1 = make_transaction(user_id="travel_u", location=london, timestamp=base)
        self.detector.evaluate(t1)

        t2 = make_transaction(
            user_id="travel_u",
            location=ny,
            timestamp=base + timedelta(minutes=5),
        )
        result = self.detector.evaluate(t2)
        assert result.final_score > 15

    def test_velocity_spike_raises_score(self):
        base = datetime(2024, 1, 1, 12, 0, 0)
        user = "velocity_test_u"
        for i in range(5):
            t = make_transaction(user_id=user, timestamp=base + timedelta(seconds=i))
            self.detector.evaluate(t)
        t6 = make_transaction(user_id=user, timestamp=base + timedelta(seconds=6))
        result = self.detector.evaluate(t6)
        assert result.final_score >= 10

    def test_final_score_in_valid_range(self):
        t = make_transaction(amount=99999.0)
        result = self.detector.evaluate(t)
        assert 0 <= result.final_score <= 100

    def test_result_has_transaction_id(self):
        t = make_transaction(transaction_id="txn_abc123")
        result = self.detector.evaluate(t)
        assert result.transaction_id == "txn_abc123"

    def test_result_has_user_id(self):
        t = make_transaction(user_id="u_xyz")
        result = self.detector.evaluate(t)
        assert result.user_id == "u_xyz"

    def test_recommended_action_populated(self):
        t = make_transaction()
        result = self.detector.evaluate(t)
        assert result.recommended_action != ""

    def test_reset_user(self):
        user = "reset_full_u"
        base = datetime(2024, 1, 1, 12, 0, 0)
        for i in range(5):
            t = make_transaction(user_id=user, timestamp=base + timedelta(seconds=i))
            self.detector.evaluate(t)
        self.detector.reset_user(user)
        t = make_transaction(user_id=user, timestamp=base + timedelta(seconds=10))
        result = self.detector.evaluate(t)
        assert result.final_score <= 30

    def test_get_user_profile_structure(self):
        t = make_transaction(user_id="profile_test")
        self.detector.evaluate(t)
        profile = self.detector.get_user_profile("profile_test")
        assert "rules_profile" in profile
        assert "amount_stats" in profile
        assert "velocity_1min" in profile
        assert "velocity_1hour" in profile

    def test_to_dict_is_serializable(self):
        import json
        t = make_transaction()
        result = self.detector.evaluate(t)
        d = result.to_dict()
        json_str = json.dumps(d)  # should not raise
        assert len(json_str) > 0
