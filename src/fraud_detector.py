import logging

from .models import Transaction, FraudResult, DetectorResult, RiskLevel
from .velocity_checker import VelocityChecker
from .geo_analyzer import GeoAnalyzer
from .amount_analyzer import AmountAnalyzer
from .rules_engine import RulesEngine
from .ml_scorer import MLScorer

logger = logging.getLogger(__name__)


class FraudDetector:
    """
    Main fraud detection engine.

    Aggregates signals from all detectors using a weighted average.
    Each detector returns a score (0-100) and a weight reflecting its
    reliability. Hard-blocking rules (blocked users/merchants) bypass
    the weighted average entirely.

    Weighted score formula:
        final_score = sum(score_i * weight_i) / sum(weight_i)

    Interview talking point: "Weighted aggregation lets us tune each
    signal's influence independently. The rules engine has the highest
    weight because it has zero false-negative rate for known bad actors.
    The ML model has lower weight because it needs warm-up data."
    """

    def __init__(
        self,
        velocity_checker: VelocityChecker | None = None,
        geo_analyzer: GeoAnalyzer | None = None,
        amount_analyzer: AmountAnalyzer | None = None,
        rules_engine: RulesEngine | None = None,
        ml_scorer: MLScorer | None = None,
    ):
        self.velocity_checker = velocity_checker or VelocityChecker()
        self.geo_analyzer = geo_analyzer or GeoAnalyzer()
        self.amount_analyzer = amount_analyzer or AmountAnalyzer()
        self.rules_engine = rules_engine or RulesEngine()
        self.ml_scorer = ml_scorer or MLScorer()

    def evaluate(self, transaction: Transaction) -> FraudResult:
        results: list[DetectorResult] = [
            self.rules_engine.check(transaction),
            self.velocity_checker.check(transaction),
            self.geo_analyzer.check(transaction),
            self.amount_analyzer.check(transaction),
            self.ml_scorer.check(transaction),
        ]

        # Hard block: if rules engine returns 100, skip weighted avg
        rules_result = results[0]
        if rules_result.score == 100.0 and "blocked_user" in rules_result.triggered_rules or \
           rules_result.score == 100.0 and "blocked_merchant" in rules_result.triggered_rules:
            logger.warning(
                "hard_block txn=%s user=%s rules=%s",
                transaction.transaction_id,
                transaction.user_id,
                rules_result.triggered_rules,
            )
            return FraudResult(
                transaction_id=transaction.transaction_id,
                user_id=transaction.user_id,
                final_score=100.0,
                risk_level=RiskLevel.BLOCK,
                detector_results=results,
            )

        total_weight = sum(r.weight for r in results)
        weighted_sum = sum(r.score * r.weight for r in results)
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        risk_level = RiskLevel.from_score(final_score)

        logger.info(
            "evaluated txn=%s user=%s score=%.1f risk=%s action=%s",
            transaction.transaction_id,
            transaction.user_id,
            final_score,
            risk_level.value,
            {RiskLevel.SAFE: "APPROVE", RiskLevel.REVIEW: "FLAG_FOR_REVIEW",
             RiskLevel.SUSPICIOUS: "REQUIRE_2FA", RiskLevel.BLOCK: "DECLINE"}[risk_level],
        )

        return FraudResult(
            transaction_id=transaction.transaction_id,
            user_id=transaction.user_id,
            final_score=final_score,
            risk_level=risk_level,
            detector_results=results,
        )

    def block_user(self, user_id: str) -> None:
        logger.warning("blocking user=%s", user_id)
        self.rules_engine.block_user(user_id)

    def block_merchant(self, merchant_id: str) -> None:
        logger.warning("blocking merchant=%s", merchant_id)
        self.rules_engine.block_merchant(merchant_id)

    def unblock_user(self, user_id: str) -> None:
        logger.info("unblocking user=%s", user_id)
        self.rules_engine.unblock_user(user_id)

    def warm_up_ml(self, transactions: list[Transaction]) -> None:
        """Pre-train the ML model with historical transaction data."""
        self.ml_scorer.warm_up(transactions)

    def reset_user(self, user_id: str) -> None:
        """Clear all state for a user (e.g. for testing or account reset)."""
        self.velocity_checker.reset_user(user_id)
        self.geo_analyzer.reset_user(user_id)
        self.amount_analyzer.reset_user(user_id)
        self.rules_engine.reset_user(user_id)

    def get_user_profile(self, user_id: str) -> dict:
        return {
            "rules_profile": self.rules_engine.get_user_profile(user_id),
            "amount_stats": self.amount_analyzer.get_user_stats(user_id),
            "velocity_1min": self.velocity_checker.get_user_transaction_count(user_id, 60),
            "velocity_1hour": self.velocity_checker.get_user_transaction_count(user_id, 3600),
        }
