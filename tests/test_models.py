import pytest
from datetime import datetime

from src.models import (
    Transaction, TransactionType, Location, FraudResult,
    DetectorResult, RiskLevel
)
from tests.conftest import make_transaction


class TestRiskLevel:
    def test_safe_score(self):
        assert RiskLevel.from_score(0) == RiskLevel.SAFE
        assert RiskLevel.from_score(30) == RiskLevel.SAFE

    def test_review_score(self):
        assert RiskLevel.from_score(31) == RiskLevel.REVIEW
        assert RiskLevel.from_score(60) == RiskLevel.REVIEW

    def test_suspicious_score(self):
        assert RiskLevel.from_score(61) == RiskLevel.SUSPICIOUS
        assert RiskLevel.from_score(80) == RiskLevel.SUSPICIOUS

    def test_block_score(self):
        assert RiskLevel.from_score(81) == RiskLevel.BLOCK
        assert RiskLevel.from_score(100) == RiskLevel.BLOCK


class TestTransaction:
    def test_valid_transaction(self):
        t = make_transaction(amount=100.0)
        assert t.amount == 100.0

    def test_negative_amount_raises(self):
        with pytest.raises(ValueError):
            make_transaction(amount=-10.0)

    def test_zero_amount_raises(self):
        with pytest.raises(ValueError):
            make_transaction(amount=0.0)

    def test_with_location(self):
        loc = Location(51.5, -0.1, "GB", "London")
        t = make_transaction(location=loc)
        assert t.location.country == "GB"


class TestFraudResult:
    def test_recommended_action_safe(self):
        result = FraudResult("t1", "u1", 20.0, RiskLevel.SAFE)
        assert result.recommended_action == "APPROVE"

    def test_recommended_action_review(self):
        result = FraudResult("t1", "u1", 45.0, RiskLevel.REVIEW)
        assert result.recommended_action == "FLAG_FOR_REVIEW"

    def test_recommended_action_suspicious(self):
        result = FraudResult("t1", "u1", 70.0, RiskLevel.SUSPICIOUS)
        assert result.recommended_action == "REQUIRE_2FA"

    def test_recommended_action_block(self):
        result = FraudResult("t1", "u1", 90.0, RiskLevel.BLOCK)
        assert result.recommended_action == "DECLINE"

    def test_to_dict_structure(self):
        dr = DetectorResult("test_detector", 50.0, 1.0, ["rule_a"])
        result = FraudResult("t1", "u1", 50.0, RiskLevel.REVIEW, [dr])
        d = result.to_dict()
        assert d["transaction_id"] == "t1"
        assert d["risk_level"] == "REVIEW"
        assert len(d["detectors"]) == 1
        assert d["detectors"][0]["name"] == "test_detector"
