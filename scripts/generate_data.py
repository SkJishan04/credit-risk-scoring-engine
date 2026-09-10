"""CLI: generate and persist a synthetic dataset to data/processed/ for offline analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from credit_risk.config import get_settings
from credit_risk.data.synthetic_generator import generate_synthetic_dataset
from credit_risk.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic MSME transaction data.")
    parser.add_argument("--n-businesses", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()

    settings = get_settings()
    n_businesses = args.n_businesses or settings.synthetic_n_businesses

    logger.info("generating_dataset", n_businesses=n_businesses)
    dataset = generate_synthetic_dataset(n_businesses=n_businesses, seed=settings.random_seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.profiles.to_csv(out_dir / "profiles.csv", index=False)
    dataset.transactions.to_csv(out_dir / "transactions.csv", index=False)

    logger.info(
        "dataset_written",
        profiles_path=str(out_dir / "profiles.csv"),
        transactions_path=str(out_dir / "transactions.csv"),
        n_transactions=len(dataset.transactions),
    )


if __name__ == "__main__":
    main()