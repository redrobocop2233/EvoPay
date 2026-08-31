"""Pydantic request and response models for EVO-PAY Blue Team API."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Actionable response decisions for fraud evaluation."""
    ALLOW = "allow"
    CHALLENGE = "challenge"
    HOLD = "hold"
    BLOCK = "block"


class TransactionContext(BaseModel):
    """Incoming transaction to evaluate for fraud risk."""
    campaign_id: str = Field(..., description="Red Team campaign identifier")
    strategy_id: Optional[str] = Field(None, description="Red Team strategy variant")
    customer_id: str = Field(..., description="Customer performing the transaction")
    transaction: dict = Field(
        ...,
        description=(
            "Raw transaction fields (amount, merchant_id, merchant_category, "
            "device_id, ip_address, location_lat, location_lon, timestamp, "
            "currency, channel)"
        ),
    )
    customer_history: Optional[list[dict]] = Field(
        None, description="Recent transaction history for this customer"
    )
    ecosystem_context: Optional[dict] = Field(
        None, description="Cross-team shared intelligence"
    )


class RiskResponse(BaseModel):
    """Blue Team fraud evaluation response."""
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated fraud probability"
    )
    detected: bool = Field(
        ..., description="Whether this transaction is flagged as fraud"
    )
    decision: Decision = Field(..., description="Actionable response decision")
    reason_codes: list[str] = Field(
        default_factory=list, description="Human-readable explanation codes"
    )
    model_scores: dict = Field(
        default_factory=lambda: {
            "tabular": 0.0,
            "anomaly": 0.0,
            "graph": 0.0,
            "temporal": 0.0,
        },
        description="Per-component model scores (private detail, helps debugging)",
    )
