"""CLI: run the full training pipeline and persist model artifacts."""

from __future__ import annotations

from credit_risk.logging_config import configure_logging, get_logger
from credit_risk.training.train_pipeline import run_training_pipeline

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    metrics = run_training_pipeline()
    logger.info("training_complete", **metrics)


if __name__ == "__main__":
    main()