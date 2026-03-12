import pytest
from src.amount_analyzer import AmountAnalyzer, WelfordStats
from tests.conftest import make_transaction


class TestWelfordStats:
    def test_single_value_mean(self):
        w = WelfordStats()
        w.update(100.0)
        assert w.mean == pytest.approx(100.0)

    def test_mean_of_sequence(self):
        w = WelfordStats()
        for v in [10, 20, 30, 40, 50]:
            w.update(v)
        assert w.mean == pytest.approx(30.0)

    def test_std_dev_known_sequence(self):
        # Sample std_dev of [2,4,4,4,5,5,7,9] = sqrt(32/7) ≈ 2.138
        w = WelfordStats()
        for v in [2, 4, 4, 4, 5, 5, 7, 9]:
            w.update(v)
        assert w.std_dev == pytest.approx(2.138, abs=0.01)

    def test_zero_std_dev_for_single_value(self):
        w = WelfordStats()
        w.update(50.0)
        assert w.std_dev == 0.0
        assert w.z_score(50.0) == 0.0

    def test_z_score_mean_is_zero(self):
        w = WelfordStats()
        for v in [10, 20, 30, 40, 50]:
            w.update(v)
        assert w.z_score(w.mean) == pytest.approx(0.0, abs=0.01)

    def test_z_score_outlier(self):
        w = WelfordStats()
        # Mix of values to establish non-zero std_dev
        for v in [10, 12, 11, 13, 10, 12, 11, 13, 10, 12]:
            w.update(float(v))
        # A value very far from mean ~11.5 should have high z-score
        z = w.z_score(100.0)
        assert z > 3.0


class TestAmountAnalyzer:
    def setup_method(self):
        self.analyzer = AmountAnalyzer()

    def test_first_transaction_no_score(self):
        t = make_transaction(amount=47.33)  # non-round, no heuristic triggers
        result = self.analyzer.check(t)
        assert result.score == 0.0

    def test_normal_amounts_no_trigger(self):
        user = "normal_user"
        for _ in range(10):
            t = make_transaction(user_id=user, amount=50.0)
            self.analyzer.check(t)
        t = make_transaction(user_id=user, amount=55.0)
        result = self.analyzer.check(t)
        assert "amount_outlier" not in result.triggered_rules

    def test_extreme_outlier_detected(self):
        user = "outlier_user"
        # Build a history of small transactions
        for _ in range(10):
            self.analyzer.check(make_transaction(user_id=user, amount=10.0))
        # Now submit a massive one
        result = self.analyzer.check(make_transaction(user_id=user, amount=10000.0))
        assert len(result.triggered_rules) > 0
        assert result.score > 0

    def test_absolute_high_amount(self):
        t = make_transaction(amount=15000.0)
        result = self.analyzer.check(t)
        assert "absolute_high_amount" in result.triggered_rules
        assert result.score >= 55

    def test_micro_transaction_flagged(self):
        t = make_transaction(amount=0.50)
        result = self.analyzer.check(t)
        assert "micro_transaction" in result.triggered_rules

    def test_just_below_threshold_flagged(self):
        t = make_transaction(amount=995.0)  # just below £1000
        result = self.analyzer.check(t)
        assert "just_below_threshold" in result.triggered_rules

    def test_round_number_bias(self):
        t = make_transaction(amount=100.0)
        result = self.analyzer.check(t)
        assert "round_number_bias" in result.triggered_rules

    def test_non_round_no_bias(self):
        t = make_transaction(amount=47.33)
        result = self.analyzer.check(t)
        assert "round_number_bias" not in result.triggered_rules

    def test_get_user_stats_empty(self):
        stats = self.analyzer.get_user_stats("no_such_user")
        assert stats["count"] == 0

    def test_get_user_stats_after_transactions(self):
        user = "stats_user"
        for _ in range(5):
            self.analyzer.check(make_transaction(user_id=user, amount=100.0))
        stats = self.analyzer.get_user_stats(user)
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(100.0)

    def test_reset_user(self):
        user = "reset_amount_user"
        for _ in range(10):
            self.analyzer.check(make_transaction(user_id=user, amount=50.0))
        self.analyzer.reset_user(user)
        stats = self.analyzer.get_user_stats(user)
        assert stats["count"] == 0

    def test_score_capped_at_100(self):
        user = "cap_user"
        for _ in range(10):
            self.analyzer.check(make_transaction(user_id=user, amount=1.0))
        result = self.analyzer.check(make_transaction(user_id=user, amount=999999.0))
        assert result.score <= 100.0

    def test_detector_name(self):
        result = self.analyzer.check(make_transaction())
        assert result.detector_name == "amount_analyzer"
