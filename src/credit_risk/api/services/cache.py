"""Thin caching layer around Redis for repeated scoring requests."""

from __future__ import annotations

import hashlib
import json

import redis

from credit_risk.api.schemas import ScoreRequestSchema, ScoreResponseSchema
from credit_risk.config import get_settings


def _cache_key(request: ScoreRequestSchema) -> str:
    payload = request.model_dump_json()
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"score:{request.business_profile.business_id}:{digest}"


class ScoreCache:
    def __init__(self, client: redis.Redis):
        self.client = client
        self.ttl = get_settings().score_cache_ttl_seconds

    def get(self, request: ScoreRequestSchema) -> ScoreResponseSchema | None:
        raw = self.client.get(_cache_key(request))
        if raw is None:
            return None
        data = json.loads(raw)
        data["cached"] = True
        return ScoreResponseSchema(**data)

    def set(self, request: ScoreRequestSchema, response: ScoreResponseSchema) -> None:
        payload = response.model_dump()
        payload["cached"] = False
        self.client.setex(_cache_key(request), self.ttl, json.dumps(payload))