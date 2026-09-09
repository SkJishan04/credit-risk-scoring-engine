"""FastAPI dependency providers: DB session, cache client, and model ensemble."""

from __future__ import annotations

from functools import lru_cache

import redis

from credit_risk.config import Settings, get_settings
from credit_risk.models.ensemble import CreditRiskEnsemble


@lru_cache
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_ensemble() -> CreditRiskEnsemble:
    settings: Settings = get_settings()
    if not settings.tgat_model_path.exists() or not settings.xgb_model_path.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python scripts/train.py` before starting the API."
        )
    return CreditRiskEnsemble.load(
        tgat_path=settings.tgat_model_path,
        xgb_path=settings.xgb_model_path,
        random_seed=settings.random_seed,
    )