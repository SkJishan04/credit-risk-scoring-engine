"""Health and readiness endpoints."""

from fastapi import APIRouter

from credit_risk.api.dependencies import get_ensemble
from credit_risk.api.schemas import HealthResponseSchema

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponseSchema)
def health_check() -> HealthResponseSchema:
    try:
        get_ensemble()
        model_loaded = True
    except RuntimeError:
        model_loaded = False
    return HealthResponseSchema(status="ok", model_loaded=model_loaded)