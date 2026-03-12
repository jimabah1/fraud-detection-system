import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

from src.fraud_detector import FraudDetector
from src.models import Transaction, TransactionType, Location

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for financial transactions.",
    version="1.0.0",
)

detector = FraudDetector()


# --- Request / Response schemas ---

class LocationSchema(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    country: str = ""
    city: str = ""


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float = Field(..., gt=0)
    currency: str = Field(default="GBP", max_length=3)
    transaction_type: str = Field(default="PAYMENT")
    merchant_category: str = Field(default="general")
    timestamp: Optional[datetime] = None
    location: Optional[LocationSchema] = None
    merchant_id: str = ""
    is_international: bool = False
    device_id: str = ""

    @field_validator("transaction_type")
    @classmethod
    def validate_type(cls, v):
        valid = {t.value for t in TransactionType}
        if v.upper() not in valid:
            raise ValueError(f"transaction_type must be one of {valid}")
        return v.upper()


class BlockRequest(BaseModel):
    user_id: Optional[str] = None
    merchant_id: Optional[str] = None


class UserProfileResponse(BaseModel):
    user_id: str
    profile: dict


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "service": "fraud-detection-api"}


@app.post("/check-transaction")
def check_transaction(req: TransactionRequest):
    """
    Evaluate a transaction for fraud risk.

    Returns a risk score (0-100), risk level, recommended action,
    and per-detector breakdown.
    """
    location = None
    if req.location:
        location = Location(
            latitude=req.location.latitude,
            longitude=req.location.longitude,
            country=req.location.country,
            city=req.location.city,
        )

    try:
        transaction = Transaction(
            transaction_id=req.transaction_id,
            user_id=req.user_id,
            amount=req.amount,
            currency=req.currency,
            transaction_type=TransactionType(req.transaction_type),
            merchant_category=req.merchant_category,
            timestamp=req.timestamp or datetime.utcnow(),
            location=location,
            merchant_id=req.merchant_id,
            is_international=req.is_international,
            device_id=req.device_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = detector.evaluate(transaction)
    return result.to_dict()


@app.post("/block")
def block_entity(req: BlockRequest):
    """Block a user or merchant from transacting."""
    if not req.user_id and not req.merchant_id:
        raise HTTPException(status_code=422, detail="Provide user_id or merchant_id")

    if req.user_id:
        detector.block_user(req.user_id)
    if req.merchant_id:
        detector.block_merchant(req.merchant_id)

    return {"status": "blocked", "user_id": req.user_id, "merchant_id": req.merchant_id}


@app.delete("/block/user/{user_id}")
def unblock_user(user_id: str):
    """Remove a user block."""
    detector.unblock_user(user_id)
    return {"status": "unblocked", "user_id": user_id}


@app.get("/user/{user_id}/profile")
def get_user_profile(user_id: str):
    """Get the fraud detection profile for a user."""
    profile = detector.get_user_profile(user_id)
    return {"user_id": user_id, "profile": profile}
