"""Request/response schemas for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from credit_risk.data.schemas import BusinessProfile, Transaction


class ScoreRequestSchema(BaseModel):
    business_profile: BusinessProfile
    transactions: list[Transaction] = Field(min_length=1)


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class ScoreResponseSchema(BaseModel):
    business_id: str
    default_probability: float
    credit_score: int
    recommended_interest_rate_pct: float
    top_contributing_factors: list[FeatureContribution]
    cached: bool = False


class HealthResponseSchema(BaseModel):
    status: str
    model_loaded: bool