"""FastAPI application entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from credit_risk.api.routers import health, scoring
from credit_risk.config import get_settings
from credit_risk.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("api_startup", env=settings.api_env)
    yield
    logger.info("api_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Algorithmic Credit Risk & Working Capital Scoring Engine",
        description=(
            "Hybrid T-GAT + XGBoost scoring API for MSME and gig-economy "
            "creditworthiness assessment using alternative transactional data."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.api_env == "development" else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix=settings.api_v1_prefix)
    app.include_router(scoring.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()