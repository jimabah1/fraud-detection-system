import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

VALID_TRANSACTION = {
    "transaction_id": "txn_test_001",
    "user_id": "user_api_test",
    "amount": 49.99,
    "currency": "GBP",
    "transaction_type": "PAYMENT",
    "merchant_category": "retail",
    "merchant_id": "merch_001",
    "is_international": False,
    "device_id": "device_001",
}


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "service" in data


class TestCheckTransactionEndpoint:
    def test_valid_transaction_returns_200(self):
        response = client.post("/check-transaction", json=VALID_TRANSACTION)
        assert response.status_code == 200

    def test_response_has_required_fields(self):
        response = client.post("/check-transaction", json=VALID_TRANSACTION)
        data = response.json()
        assert "transaction_id" in data
        assert "final_score" in data
        assert "risk_level" in data
        assert "recommended_action" in data
        assert "detectors" in data

    def test_score_in_valid_range(self):
        response = client.post("/check-transaction", json=VALID_TRANSACTION)
        data = response.json()
        assert 0 <= data["final_score"] <= 100

    def test_risk_level_is_valid(self):
        response = client.post("/check-transaction", json=VALID_TRANSACTION)
        data = response.json()
        assert data["risk_level"] in {"SAFE", "REVIEW", "SUSPICIOUS", "BLOCK"}

    def test_five_detectors_in_response(self):
        response = client.post("/check-transaction", json=VALID_TRANSACTION)
        data = response.json()
        names = {d["name"] for d in data["detectors"]}
        assert names == {"rules_engine", "velocity_checker", "geo_analyzer", "amount_analyzer", "ml_scorer"}

    def test_transaction_id_echoed_back(self):
        response = client.post("/check-transaction", json=VALID_TRANSACTION)
        assert response.json()["transaction_id"] == "txn_test_001"

    def test_with_location(self):
        payload = {**VALID_TRANSACTION, "location": {"latitude": 51.5, "longitude": -0.1, "country": "GB"}}
        response = client.post("/check-transaction", json=payload)
        assert response.status_code == 200

    def test_zero_amount_rejected(self):
        payload = {**VALID_TRANSACTION, "amount": 0}
        response = client.post("/check-transaction", json=payload)
        assert response.status_code == 422

    def test_negative_amount_rejected(self):
        payload = {**VALID_TRANSACTION, "amount": -10.0}
        response = client.post("/check-transaction", json=payload)
        assert response.status_code == 422

    def test_invalid_transaction_type_rejected(self):
        payload = {**VALID_TRANSACTION, "transaction_type": "FAKE_TYPE"}
        response = client.post("/check-transaction", json=payload)
        assert response.status_code == 422

    def test_missing_required_fields_rejected(self):
        response = client.post("/check-transaction", json={"amount": 50.0})
        assert response.status_code == 422

    def test_invalid_latitude_rejected(self):
        payload = {**VALID_TRANSACTION, "location": {"latitude": 999, "longitude": 0}}
        response = client.post("/check-transaction", json=payload)
        assert response.status_code == 422


class TestBlockEndpoint:
    def test_block_user_returns_200(self):
        response = client.post("/block", json={"user_id": "bad_actor_1"})
        assert response.status_code == 200

    def test_block_merchant_returns_200(self):
        response = client.post("/block", json={"merchant_id": "bad_merch_1"})
        assert response.status_code == 200

    def test_blocked_user_transaction_is_declined(self):
        client.post("/block", json={"user_id": "blocked_api_user"})
        payload = {**VALID_TRANSACTION, "user_id": "blocked_api_user", "transaction_id": "txn_blocked_1"}
        response = client.post("/check-transaction", json=payload)
        data = response.json()
        assert data["risk_level"] == "BLOCK"
        assert data["recommended_action"] == "DECLINE"

    def test_block_with_no_target_rejected(self):
        response = client.post("/block", json={})
        assert response.status_code == 422


class TestUnblockEndpoint:
    def test_unblock_user_returns_200(self):
        client.post("/block", json={"user_id": "temp_blocked_user"})
        response = client.delete("/block/user/temp_blocked_user")
        assert response.status_code == 200

    def test_unblocked_user_can_transact(self):
        client.post("/block", json={"user_id": "unblock_test_user"})
        client.delete("/block/user/unblock_test_user")
        payload = {**VALID_TRANSACTION, "user_id": "unblock_test_user", "transaction_id": "txn_unblocked"}
        response = client.post("/check-transaction", json=payload)
        assert response.json()["risk_level"] != "BLOCK"


class TestUserProfileEndpoint:
    def test_profile_returns_200(self):
        payload = {**VALID_TRANSACTION, "user_id": "profile_api_user", "transaction_id": "txn_profile_1"}
        client.post("/check-transaction", json=payload)
        response = client.get("/user/profile_api_user/profile")
        assert response.status_code == 200

    def test_profile_has_expected_keys(self):
        payload = {**VALID_TRANSACTION, "user_id": "profile_api_user2", "transaction_id": "txn_profile_2"}
        client.post("/check-transaction", json=payload)
        response = client.get("/user/profile_api_user2/profile")
        data = response.json()
        assert "user_id" in data
        assert "profile" in data
        profile = data["profile"]
        assert "velocity_1min" in profile
        assert "velocity_1hour" in profile
        assert "amount_stats" in profile
