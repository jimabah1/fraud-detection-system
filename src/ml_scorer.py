import threading
from datetime import datetime

from .models import Transaction, DetectorResult

try:
    from sklearn.ensemble import IsolationForest
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _extract_features(transaction: Transaction, hour_of_day: int, day_of_week: int) -> list[float]:
    """
    Convert a transaction into a numeric feature vector for ML scoring.

    Features chosen to be interpretable and interview-explainable:
    - amount (log-transformed to handle skewed distribution)
    - hour of day (0-23)
    - day of week (0=Mon, 6=Sun)
    - is_international (binary)
    - merchant_category_hash (stable numeric encoding)
    - transaction_type_id
    """
    import math

    type_map = {"PAYMENT": 0, "TRANSFER": 1, "WITHDRAWAL": 2, "DEPOSIT": 3, "CARD_PURCHASE": 4}
    type_id = type_map.get(transaction.transaction_type.value, 0)

    log_amount = math.log1p(transaction.amount)
    category_hash = abs(hash(transaction.merchant_category)) % 1000

    return [
        log_amount,
        hour_of_day,
        day_of_week,
        float(transaction.is_international),
        float(type_id),
        float(category_hash),
    ]


class MLScorer:
    """
    Anomaly detection using scikit-learn's Isolation Forest.

    Isolation Forest works by randomly isolating observations - anomalies
    (fraud) are isolated in fewer splits than normal transactions.
    Unsupervised: no labeled fraud data required.

    Falls back to rule-based scoring if sklearn is not installed.

    Interview talking point: "We use unsupervised ML because labeled fraud
    datasets are rare and expensive. Isolation Forest gives us a baseline
    anomaly score that complements our deterministic rules."
    """

    MIN_SAMPLES_TO_TRAIN = 20
    RETRAIN_INTERVAL = 100  # retrain every N new transactions

    def __init__(self, contamination: float = 0.05):
        """
        contamination: expected fraction of anomalies in training data (0.05 = 5%).
        """
        self._contamination = contamination
        self._model = None
        self._training_data: list[list[float]] = []
        self._transactions_since_retrain = 0
        self._lock = threading.Lock()

    def _build_features(self, transaction: Transaction) -> list[float]:
        hour = transaction.timestamp.hour
        dow = transaction.timestamp.weekday()
        return _extract_features(transaction, hour, dow)

    def _train(self) -> None:
        if not SKLEARN_AVAILABLE or len(self._training_data) < self.MIN_SAMPLES_TO_TRAIN:
            return
        X = np.array(self._training_data)
        self._model = IsolationForest(
            contamination=self._contamination,
            n_estimators=100,
            random_state=42,
        )
        self._model.fit(X)
        self._transactions_since_retrain = 0

    def check(self, transaction: Transaction) -> DetectorResult:
        triggered = []
        score = 0.0
        metadata = {}

        features = self._build_features(transaction)

        with self._lock:
            self._training_data.append(features)
            self._transactions_since_retrain += 1

            if self._transactions_since_retrain >= self.RETRAIN_INTERVAL:
                self._train()

            if not SKLEARN_AVAILABLE:
                metadata["note"] = "sklearn_not_installed_fallback_active"
                return DetectorResult(
                    detector_name="ml_scorer",
                    score=0.0,
                    weight=0.8,
                    triggered_rules=[],
                    metadata=metadata,
                )

            if self._model is None:
                if len(self._training_data) >= self.MIN_SAMPLES_TO_TRAIN:
                    self._train()
                else:
                    metadata["note"] = f"warming_up_{len(self._training_data)}_of_{self.MIN_SAMPLES_TO_TRAIN}_samples"
                    return DetectorResult(
                        detector_name="ml_scorer",
                        score=0.0,
                        weight=0.8,
                        triggered_rules=[],
                        metadata=metadata,
                    )

            X = np.array([features])
            # decision_function returns negative scores for anomalies
            raw_score = self._model.decision_function(X)[0]
            prediction = self._model.predict(X)[0]  # -1 = anomaly, 1 = normal

            metadata["raw_isolation_score"] = round(float(raw_score), 4)
            metadata["prediction"] = "anomaly" if prediction == -1 else "normal"

            if prediction == -1:
                # Map raw score [-0.5, 0] to fraud score [100, 50]
                normalized = max(0.0, min(1.0, (-raw_score) * 2))
                score = 50.0 + normalized * 50.0
                triggered.append("isolation_forest_anomaly")

        return DetectorResult(
            detector_name="ml_scorer",
            score=min(score, 100.0),
            weight=0.8,
            triggered_rules=triggered,
            metadata=metadata,
        )

    def warm_up(self, transactions: list[Transaction]) -> None:
        """Pre-train the model with historical transactions."""
        with self._lock:
            for t in transactions:
                self._training_data.append(self._build_features(t))
            if len(self._training_data) >= self.MIN_SAMPLES_TO_TRAIN:
                self._train()

    @property
    def is_trained(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def training_sample_count(self) -> int:
        with self._lock:
            return len(self._training_data)
