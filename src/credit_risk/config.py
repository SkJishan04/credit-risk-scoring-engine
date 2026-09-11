"""Centralized, environment-driven configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    api_env: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    # Database
    database_url: str = "postgresql+psycopg://credit_user:credit_pass@localhost:5432/credit_risk_db"

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    score_cache_ttl_seconds: int = 300

    # Model artifacts
    model_artifact_dir: Path = Path("./artifacts")
    tgat_model_filename: str = "tgat_embedder.pt"
    xgb_model_filename: str = "xgb_scorer.json"

    # Experiment tracking
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "credit-risk-scoring"

    # Training
    random_seed: int = 42
    synthetic_n_businesses: int = 4000

    @property
    def tgat_model_path(self) -> Path:
        return self.model_artifact_dir / self.tgat_model_filename

    @property
    def xgb_model_path(self) -> Path:
        return self.model_artifact_dir / self.xgb_model_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()