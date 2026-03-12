import pytest
from datetime import datetime

from src.ml_scorer import MLScorer, _extract_features
from src.models import TransactionType
from tests.conftest import make_transaction


class TestFeatureExtraction:
    def test_feature_vector_length(self):
        t = make_transaction()
        features = _extract_features(t, 12, 0)
        assert len(features) == 6

    def test_log_transform_applied(self):
        import math
        t = make_transaction(amount=100.0)
        features = _extract_features(t, 12, 0)
        assert features[0] == pytest.approx(math.log1p(100.0))

    def test_international_flag(self):
        t_dom = make_transaction(is_international=False)
        t_int = make_transaction(is_international=True)
        f_dom = _extract_features(t_dom, 12, 0)
        f_int = _extract_features(t_int, 12, 0)
        assert f_dom[3] == 0.0
        assert f_int[3] == 1.0


class TestMLScorer:
    def setup_method(self):
        self.scorer = MLScorer()

    def test_returns_detector_result(self):
        t = make_transaction()
        result = self.scorer.check(t)
        assert result.detector_name == "ml_scorer"

    def test_warming_up_returns_zero_score(self):
        t = make_transaction()
        result = self.scorer.check(t)
        assert result.score == 0.0
        assert "warming_up" in result.metadata.get("note", "")

    def test_warm_up_trains_model(self):
        transactions = [make_transaction(amount=float(i * 10 + 10)) for i in range(25)]
        self.scorer.warm_up(transactions)
        assert self.scorer.is_trained

    def test_trained_model_returns_score(self):
        transactions = [make_transaction(amount=float(i * 5 + 10)) for i in range(25)]
        self.scorer.warm_up(transactions)
        t = make_transaction(amount=50.0)
        result = self.scorer.check(t)
        assert 0.0 <= result.score <= 100.0

    def test_score_in_valid_range(self):
        transactions = [make_transaction(amount=50.0) for _ in range(25)]
        self.scorer.warm_up(transactions)
        t = make_transaction(amount=999999.0)
        result = self.scorer.check(t)
        assert 0.0 <= result.score <= 100.0

    def test_training_sample_count_increments(self):
        initial = self.scorer.training_sample_count
        self.scorer.check(make_transaction())
        assert self.scorer.training_sample_count == initial + 1

    def test_weight_is_less_than_rules(self):
        result = self.scorer.check(make_transaction())
        assert result.weight < 2.0  # rules engine has 2.0
