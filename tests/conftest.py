from datetime import datetime, timedelta
import pytest

from src.models import Transaction, TransactionType, Location


def make_transaction(
    user_id="user_1",
    amount=50.0,
    transaction_type=TransactionType.PAYMENT,
    merchant_category="retail",
    timestamp=None,
    location=None,
    is_international=False,
    merchant_id="merch_1",
    device_id="device_1",
    transaction_id=None,
    currency="GBP",
) -> Transaction:
    return Transaction(
        transaction_id=transaction_id or f"txn_{datetime.utcnow().timestamp()}",
        user_id=user_id,
        amount=amount,
        currency=currency,
        transaction_type=transaction_type,
        merchant_category=merchant_category,
        timestamp=timestamp or datetime.utcnow(),
        location=location,
        merchant_id=merchant_id,
        is_international=is_international,
        device_id=device_id,
    )


def make_location(lat=51.5, lon=-0.1, country="GB", city="London") -> Location:
    return Location(latitude=lat, longitude=lon, country=country, city=city)


@pytest.fixture
def london():
    return make_location(51.5074, -0.1278, "GB", "London")


@pytest.fixture
def new_york():
    return make_location(40.7128, -74.0060, "US", "New York")


@pytest.fixture
def sydney():
    return make_location(-33.8688, 151.2093, "AU", "Sydney")
