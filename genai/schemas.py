"""
Strict schemas for everything GenAI is allowed to produce.

Per the design brief: the LLM's output is never trusted directly. Every
field is validated against the exact vocabulary red_team.py's GenomeCodec
already understands (pattern "type" values, realistic magnitude ranges) -
an invalid or out-of-vocabulary response is a validation error, not
something silently coerced into looking valid.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# --- pattern sub-schemas: same shape red_team.py's GenomeCodec decodes to ---

class TemporalPattern(BaseModel):
    type: Literal["normal", "shift_to_offhours"]
    magnitude: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("magnitude")
    @classmethod
    def magnitude_required_if_active(cls, v, info):
        if info.data.get("type") == "shift_to_offhours" and v is None:
            raise ValueError("shift_to_offhours requires a magnitude")
        return v


class AmountPattern(BaseModel):
    type: Literal["normal", "gradual_drift", "abrupt_spike", "card_testing"]
    magnitude: Optional[float] = Field(default=None, ge=0.0, le=12.0)
    duration: Optional[int] = Field(default=None, ge=2, le=10)
    probe_count: Optional[int] = Field(default=None, ge=2, le=6)
    probe_amount: Optional[float] = Field(default=None, ge=0.1, le=10.0)
    final_magnitude: Optional[float] = Field(default=None, ge=1.0, le=10.0)

    @model_validator(mode="after")
    def active_fields_required(self):
        if self.type == "gradual_drift" and self.duration is None:
            raise ValueError("gradual_drift requires duration")
        if self.type == "card_testing" and (self.probe_count is None or self.probe_amount is None or self.final_magnitude is None):
            raise ValueError("card_testing requires probe_count, probe_amount and final_magnitude")
        if self.type == "abrupt_spike" and self.magnitude is None:
            raise ValueError("abrupt_spike requires magnitude")
        return self


class DevicePattern(BaseModel):
    type: Literal["normal", "switch"]


class GeographicPattern(BaseModel):
    type: Literal["normal", "distribution_shift"]
    magnitude: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def active_fields_required(self):
        if self.type == "distribution_shift" and self.magnitude is None:
            raise ValueError("distribution_shift requires a magnitude")
        return self


class MerchantPattern(BaseModel):
    type: Literal["normal", "category_drift"]


class VelocityPattern(BaseModel):
    type: Literal["normal", "burst"]
    count: Optional[int] = Field(default=None, ge=2, le=6)
    interval_minutes: Optional[int] = Field(default=None, ge=1, le=15)

    @model_validator(mode="after")
    def active_fields_required(self):
        if self.type == "burst" and (self.count is None or self.interval_minutes is None):
            raise ValueError("burst requires count and interval_minutes")
        return self


class CoordinationPattern(BaseModel):
    type: Literal["normal", "multi_account"]


# --- top-level hypothesis ---

class AttackHypothesis(BaseModel):
    """One GenAI-proposed attack idea. `modality` decides whether it can
    become a real AttackGenome (transaction_simulatable) or is filed as
    research-only (identified but this system has no simulator for it -
    per IDENTIFY.md's own coverage matrix, e.g. deepfake/voice/document
    attacks). Pattern fields are optional so a research_only hypothesis
    doesn't need to force-fit a transaction pattern it doesn't have."""

    name: str = Field(min_length=3, max_length=80)
    family: str = Field(min_length=3, max_length=60)
    objective: str = Field(min_length=10, max_length=300)
    rationale: str = Field(min_length=10, max_length=500,
                            description="Why this is plausible, grounded in real fraud mechanics")
    modality: Literal["transaction_simulatable", "research_only"]

    temporal_pattern: Optional[TemporalPattern] = None
    amount_pattern: Optional[AmountPattern] = None
    device_pattern: Optional[DevicePattern] = None
    geographic_pattern: Optional[GeographicPattern] = None
    merchant_pattern: Optional[MerchantPattern] = None
    velocity_pattern: Optional[VelocityPattern] = None
    coordination_pattern: Optional[CoordinationPattern] = None

    @model_validator(mode="after")
    def validate_simulatable_patterns(self):
        if self.modality == "transaction_simulatable":
            patterns = (
                self.temporal_pattern, self.amount_pattern, self.device_pattern,
                self.geographic_pattern, self.merchant_pattern,
                self.velocity_pattern, self.coordination_pattern,
            )
            if not any(p is not None and p.type != "normal" for p in patterns):
                raise ValueError("transaction_simulatable hypothesis must contain at least one active pattern")
        return self


class RecommendedMutation(BaseModel):
    dimension: Literal["amount", "temporal", "device", "geographic",
                        "merchant", "velocity", "coordination"]
    direction: Literal["increase", "decrease", "remove", "add"]
    rationale: str = Field(min_length=5, max_length=200)


class AttackAutopsy(BaseModel):
    """Post-detection explanation. Never influences the actual risk score -
    Blue's decision is already final by the time this runs. This exists to
    make evolution's next mutation step interpretable, and for the
    dashboard's explanation panel."""

    strategy_id: str
    blue_risk_score: float = Field(ge=0.0, le=1.0)
    detected: bool
    exploited_weaknesses: list[str] = Field(default_factory=list, max_length=10)
    weakest_signal: Optional[str] = None
    recommended_mutations: list[RecommendedMutation] = Field(default_factory=list, max_length=5)
    explanation: str = Field(min_length=10, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
