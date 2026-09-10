"""
Integration tests for the scoring API.

The trained ensemble is a real, filesystem-backed artifact, so these tests
train a tiny model on a small synthetic dataset in a temp directory and
exercise the FastAPI app end-to-end against it (no network calls, no
external services required besides an in-memory fake for Redis).
"""

from __future__ import annotations

import fakeredis
import pytest
from fastapi.testclient import TestClient

from credit_risk.api import dependencies
from credit_risk.config import get_settings
from credit_risk.training.train_pipeline import run_training_pipeline


@pytest.fixture(scope="module")
def trained_settings(tmp_path_factory):
    get_settings.cache_clear()
    settings = get_settings()
    settings.model_artifact_dir = tmp_path_factory.mktemp("artifacts")
    settings.synthetic_n_businesses = 80
    run_training_pipeline(settings)
    return settings


@pytest.fixture
def client(trained_settings, monkeypatch):
    dependencies.get_ensemble.cache_clear()
    dependencies.get_redis_client.cache_clear()

    monkeypatch.setattr(dependencies, "get_settings", lambda: trained_settings)
    monkeypatch.setattr(dependencies, "get_redis_client", lambda: fakeredis.FakeRedis(decode_responses=True))

    from credit_risk.api.main import create_app

    app = create_app()
    return TestClient(app)


def _sample_payload():
    return {
        "business_profile": {
            "business_id": "biz_test_001",
            "sector": "retail",
            "months_active": 24,
            "declared_monthly_revenue": 50000.0,
            "n_active_vendors": 3,
            "n_active_buyers": 5,
        },
        "transactions": [
            {
                "transaction_id": f"tx_{i}",
                "business_id": "biz_test_001",
                "counterparty_id": f"vendor_{i % 3}",
                "amount": 1200.0 + i * 10,
                "direction": "outflow" if i % 2 == 0 else "inflow",
                "transaction_date": "2024-01-01",
                "invoice_due_date": "2024-01-15",
                "settled_on_time": True,
            }
            for i in range(20)
        ],
    }


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_scoring_endpoint_returns_valid_score(client):
    response = client.post("/api/v1/scoring", json=_sample_payload())
    assert response.status_code == 200
    body = response.json()

    assert body["business_id"] == "biz_test_001"
    assert 0.0 <= body["default_probability"] <= 1.0
    assert 300 <= body["credit_score"] <= 900
    assert body["recommended_interest_rate_pct"] > 0
    assert len(body["top_contributing_factors"]) > 0


def test_scoring_endpoint_rejects_mismatched_business_id(client):
    payload = _sample_payload()
    payload["transactions"][0]["business_id"] = "someone_else"
    response = client.post("/api/v1/scoring", json=payload)
    assert response.status_code == 422


def test_scoring_endpoint_uses_cache_on_repeat_request(client):
    payload = _sample_payload()
    first = client.post("/api/v1/scoring", json=payload)
    second = client.post("/api/v1/scoring", json=payload)

    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["cached"] is True