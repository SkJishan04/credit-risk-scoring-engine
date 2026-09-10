"""Unit tests for tabular feature engineering."""

from __future__ import annotations

from credit_risk.features.engineering import TABULAR_FEATURE_COLUMNS, build_tabular_features


def test_build_tabular_features_shape(sample_transactions_df, sample_profiles_df):
    features = build_tabular_features(sample_transactions_df, sample_profiles_df)

    assert "business_id" in features.columns
    for col in TABULAR_FEATURE_COLUMNS:
        assert col in features.columns
    assert len(features) == sample_profiles_df["business_id"].nunique()


def test_features_contain_no_nulls(sample_transactions_df, sample_profiles_df):
    features = build_tabular_features(sample_transactions_df, sample_profiles_df)
    assert features[TABULAR_FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_vendor_concentration_is_bounded(sample_transactions_df, sample_profiles_df):
    features = build_tabular_features(sample_transactions_df, sample_profiles_df)
    assert features["vendor_concentration_hhi"].between(0, 1).all()
    assert features["buyer_concentration_hhi"].between(0, 1).all()


def test_single_counterparty_has_max_concentration():
    import pandas as pd

    transactions = pd.DataFrame(
        [
            {
                "business_id": "biz_x",
                "counterparty_id": "only_vendor",
                "amount": 100.0,
                "direction": "outflow",
                "transaction_date": "2024-01-01",
                "invoice_due_date": "2024-01-10",
                "settled_on_time": True,
            }
        ]
    )
    profiles = pd.DataFrame(
        [
            {
                "business_id": "biz_x",
                "sector": "retail",
                "months_active": 12,
                "declared_monthly_revenue": 1000.0,
                "n_active_vendors": 1,
                "n_active_buyers": 0,
            }
        ]
    )
    features = build_tabular_features(transactions, profiles)
    assert features.loc[0, "vendor_concentration_hhi"] == 1.0