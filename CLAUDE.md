# Fraud Detection System — Claude Context

## Project
Final project in a fintech portfolio (job search). Demonstrates Rules + Statistics + ML fraud detection.
Target companies: Revolut and similar fintech.

## Run tests
```bash
source venv/Scripts/activate && python -m pytest tests/ -v
```

## Current status
- 118/118 tests passing
- All 5 detectors implemented: rules_engine, velocity_checker, geo_analyzer, amount_analyzer, ml_scorer
- FastAPI REST layer with 5 endpoints
- API integration tests in tests/test_api.py
- Basic logging in src/fraud_detector.py and api/app.py

## Architecture
Weighted aggregation across 5 detectors:
- rules_engine (weight 2.0) — deterministic rules, hard blocks
- velocity_checker (weight 1.5) — sliding window, deque-based
- geo_analyzer (weight 1.2) — Haversine impossible travel detection
- amount_analyzer (weight 1.0) — Welford's online algorithm for z-score outliers
- ml_scorer (weight 0.8) — Isolation Forest, cold-starts at 20 samples

## Known limitations (by design — portfolio scope)
- All state in-memory (not distributed, lost on restart)
- ML model not persisted across restarts
- No auth/rate limiting on API
- No Redis/database persistence
