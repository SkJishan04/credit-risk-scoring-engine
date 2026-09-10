"""Scoring endpoints: submit transactional data, receive a dynamic credit score."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from credit_risk.api.dependencies import get_ensemble, get_redis_client
from credit_risk.api.schemas import ScoreRequestSchema, ScoreResponseSchema
from credit_risk.api.services.cache import ScoreCache
from credit_risk.api.services.scoring_service import ScoringService
from credit_risk.db.repository import BusinessRepository
from credit_risk.db.session import get_db
from credit_risk.logging_config import get_logger
from credit_risk.models.ensemble import CreditRiskEnsemble

router = APIRouter(prefix="/scoring", tags=["scoring"])
logger = get_logger(__name__)


@router.post("", response_model=ScoreResponseSchema, status_code=status.HTTP_200_OK)
def score_business(
    request: ScoreRequestSchema,
    db: Session = Depends(get_db),
    ensemble: CreditRiskEnsemble = Depends(get_ensemble),
) -> ScoreResponseSchema:
    if request.business_profile.business_id != request.transactions[0].business_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="business_profile.business_id must match transactions[*].business_id",
        )

    cache = ScoreCache(get_redis_client())
    cached_response = cache.get(request)
    if cached_response is not None:
        logger.info("scoring_cache_hit", business_id=request.business_profile.business_id)
        return cached_response

    try:
        service = ScoringService(ensemble=ensemble, db=db)
        response = service.score(request)
    except Exception as exc:  # noqa: BLE001
        logger.error("scoring_failed", error=str(exc), business_id=request.business_profile.business_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Scoring failed."
        ) from exc

    cache.set(request, response)
    logger.info(
        "scoring_success",
        business_id=response.business_id,
        credit_score=response.credit_score,
        default_probability=response.default_probability,
    )
    return response


@router.get("/{business_id}/latest", response_model=ScoreResponseSchema | None)
def get_latest_score(business_id: str, db: Session = Depends(get_db)) -> ScoreResponseSchema | None:
    repo = BusinessRepository(db)
    record = repo.get_latest_score(business_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No score found.")
    return ScoreResponseSchema(
        business_id=record.business_id,
        default_probability=record.default_probability,
        credit_score=record.credit_score,
        recommended_interest_rate_pct=record.recommended_interest_rate_pct,
        top_contributing_factors=[],
        cached=False,
    )