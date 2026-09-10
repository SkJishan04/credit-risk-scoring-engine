"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.data.synthetic_generator import generate_synthetic_dataset


@pytest.fixture(scope="session")
def small_synthetic_dataset():
    return generate_synthetic_dataset(n_businesses=60, seed=7)


@pytest.fixture
def sample_transactions_df(small_synthetic_dataset) -> pd.DataFrame:
    return small_synthetic_dataset.transactions


@pytest.fixture
def sample_profiles_df(small_synthetic_dataset) -> pd.DataFrame:
    return small_synthetic_dataset.profiles