# Fraud Detection System

A real-time fraud detection engine with rule-based, statistical, and ML-based signals. Built as Project #6 in a fintech engineering portfolio.

## Features

| Signal | Technique | Complexity |
|--------|-----------|------------|
| Transaction velocity | Sliding window with `collections.deque` | O(1) amortized |
| Geographic anomaly | Haversine formula + impossible travel | O(1) per check |
| Amount outlier | Welford's online mean/variance (z-score) | O(1) space |
| Rule-based checks | Blacklists, thresholds, category heuristics | O(1) |
| ML anomaly score | Isolation Forest (scikit-learn) | O(n_trees) |

## Risk Levels

| Score | Level | Action |
|-------|-------|--------|
| 0 – 30 | SAFE | APPROVE |
| 31 – 60 | REVIEW | FLAG_FOR_REVIEW |
| 61 – 80 | SUSPICIOUS | REQUIRE_2FA |
| 81 – 100 | BLOCK | DECLINE |

## Project Structure

```
fraud-detection-system/
├── src/
│   ├── models.py           # Transaction, FraudResult, RiskLevel
│   ├── velocity_checker.py # Sliding window velocity detection
│   ├── geo_analyzer.py     # Geographic impossibility detection
│   ├── amount_analyzer.py  # Statistical amount profiling
│   ├── rules_engine.py     # Deterministic rule-based checks
│   ├── ml_scorer.py        # Isolation Forest anomaly detection
│   └── fraud_detector.py   # Aggregator - weighted scoring engine
├── api/
│   └── app.py              # FastAPI REST API
├── tests/                  # 60+ pytest tests, 100% pass rate
├── Dockerfile
└── docker-compose.yml
```

## Quick Start

### Local
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run API
uvicorn api.app:app --reload

# Run tests
pytest tests/ -v --cov=src
```

### Docker
```bash
docker-compose up --build
```

## API Endpoints

### `POST /check-transaction`
Evaluate a transaction for fraud risk.

```bash
curl -X POST http://localhost:8000/check-transaction \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "user_id": "user_123",
    "amount": 999.50,
    "currency": "GBP",
    "transaction_type": "PAYMENT",
    "merchant_category": "gambling",
    "is_international": false,
    "device_id": "device_abc"
  }'
```

Response:
```json
{
  "transaction_id": "txn_001",
  "user_id": "user_123",
  "final_score": 42.5,
  "risk_level": "REVIEW",
  "recommended_action": "FLAG_FOR_REVIEW",
  "detectors": [
    {"name": "rules_engine", "score": 55.0, "rules_triggered": ["high_risk_merchant_category", "just_below_threshold"]},
    {"name": "velocity_checker", "score": 0.0, "rules_triggered": []},
    ...
  ]
}
```

### `POST /block`
Block a user or merchant.

### `DELETE /block/user/{user_id}`
Remove a user block.

### `GET /user/{user_id}/profile`
Get the fraud detection profile for a user.

## Design Decisions

**Why Welford's algorithm?** Rolling z-score without storing transaction history. O(1) space per user vs O(n) for naive approaches — critical at scale.

**Why Isolation Forest?** Unsupervised — no labeled fraud data needed. Works on cold-start. Complements rule-based signals rather than replacing them.

**Why weighted aggregation?** Each detector has different reliability. The rules engine (weight 2.0) has zero false-negative rate for known bad actors. The ML model (weight 0.8) has lower weight during warm-up.

**Why `collections.deque` for velocity?** The sliding window needs O(1) pop from the left. Python lists are O(n) for this — deque makes it O(1).

## Interview Talking Points

- **Scaling**: Stateless design means horizontal scaling behind a load balancer. User state could move to Redis for distributed deployments.
- **Concurrency**: All user state protected by `threading.Lock` — safe for multi-threaded WSGI/ASGI servers.
- **False positive tuning**: Isolation Forest `contamination` parameter is configurable. Rules thresholds are externalized as constants.
- **Explainability**: Every decision includes `triggered_rules` — a full audit trail of why a transaction was flagged.
