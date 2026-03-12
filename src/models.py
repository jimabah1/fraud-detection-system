from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    SAFE = "SAFE"
    REVIEW = "REVIEW"
    SUSPICIOUS = "SUSPICIOUS"
    BLOCK = "BLOCK"

    @staticmethod
    def from_score(score: float) -> "RiskLevel":
        if score <= 30:
            return RiskLevel.SAFE
        elif score <= 60:
            return RiskLevel.REVIEW
        elif score <= 80:
            return RiskLevel.SUSPICIOUS
        else:
            return RiskLevel.BLOCK


class TransactionType(Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    WITHDRAWAL = "WITHDRAWAL"
    DEPOSIT = "DEPOSIT"
    CARD_PURCHASE = "CARD_PURCHASE"


@dataclass
class Location:
    latitude: float
    longitude: float
    country: str = ""
    city: str = ""


@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    amount: float
    currency: str
    transaction_type: TransactionType
    merchant_category: str
    timestamp: datetime
    location: Optional[Location] = None
    merchant_id: str = ""
    is_international: bool = False
    device_id: str = ""

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError(f"Transaction amount must be positive, got {self.amount}")


@dataclass
class DetectorResult:
    detector_name: str
    score: float          # 0-100
    weight: float         # relative importance
    triggered_rules: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class FraudResult:
    transaction_id: str
    user_id: str
    final_score: float
    risk_level: RiskLevel
    detector_results: list[DetectorResult] = field(default_factory=list)
    recommended_action: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        actions = {
            RiskLevel.SAFE: "APPROVE",
            RiskLevel.REVIEW: "FLAG_FOR_REVIEW",
            RiskLevel.SUSPICIOUS: "REQUIRE_2FA",
            RiskLevel.BLOCK: "DECLINE",
        }
        if not self.recommended_action:
            self.recommended_action = actions[self.risk_level]

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "final_score": round(self.final_score, 2),
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action,
            "detectors": [
                {
                    "name": d.detector_name,
                    "score": round(d.score, 2),
                    "rules_triggered": d.triggered_rules,
                }
                for d in self.detector_results
            ],
            "timestamp": self.timestamp.isoformat(),
        }
