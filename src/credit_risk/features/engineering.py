"""Hand-engineered tabular features from raw transaction logs."""

from __future__ import annotations

import numpy as np
import pandas as pd

TABULAR_FEATURE_COLUMNS = [
    "n_transactions",
    "mean_amount",
    "std_amount",
    "inflow_outflow_ratio",
    "mean_settlement_delay_days",
    "std_settlement_delay_days",
    "pct_settled_on_time",
    "mean_interval_days",
    "std_interval_days",
    "vendor_concentration_hhi",
    "buyer_concentration_hhi",
    "months_active",
    "declared_monthly_revenue",
    "n_active_vendors",
    "n_active_buyers",
    "revenue_to_txn_volume_ratio",
]


def _herfindahl_index(amounts_by_counterparty: pd.Series) -> float:
    """Concentration index: 1.0 = single counterparty, ->0 = fully diversified."""
    if amounts_by_counterparty.empty or amounts_by_counterparty.sum() == 0:
        return 0.0
    shares = amounts_by_counterparty / amounts_by_counterparty.sum()
    return float((shares**2).sum())


def build_tabular_features(
    transactions: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate raw transaction rows into one feature vector per business.

    transactions: columns matching credit_risk.data.schemas.Transaction
    profiles: columns matching credit_risk.data.schemas.BusinessProfile
    """
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    transactions["invoice_due_date"] = pd.to_datetime(transactions["invoice_due_date"])

    transactions["settlement_delay_days"] = (
        transactions["transaction_date"] - transactions["invoice_due_date"]
    ).dt.days

    records = []
    for business_id, group in transactions.groupby("business_id"):
        group = group.sort_values("transaction_date")
        intervals = group["transaction_date"].diff().dt.days.dropna()

        inflow_total = group.loc[group["direction"] == "inflow", "amount"].sum()
        outflow_total = group.loc[group["direction"] == "outflow", "amount"].sum()

        vendor_amounts = group.loc[group["direction"] == "outflow"].groupby("counterparty_id")[
            "amount"
        ].sum()
        buyer_amounts = group.loc[group["direction"] == "inflow"].groupby("counterparty_id")[
            "amount"
        ].sum()

        records.append(
            {
                "business_id": business_id,
                "n_transactions": len(group),
                "mean_amount": group["amount"].mean(),
                "std_amount": group["amount"].std(ddof=0) or 0.0,
                "inflow_outflow_ratio": float(inflow_total / outflow_total)
                if outflow_total > 0
                else float(inflow_total > 0),
                "mean_settlement_delay_days": group["settlement_delay_days"].mean(),
                "std_settlement_delay_days": group["settlement_delay_days"].std(ddof=0) or 0.0,
                "pct_settled_on_time": group["settled_on_time"].mean(),
                "mean_interval_days": intervals.mean() if not intervals.empty else 0.0,
                "std_interval_days": intervals.std(ddof=0) if not intervals.empty else 0.0,
                "vendor_concentration_hhi": _herfindahl_index(vendor_amounts),
                "buyer_concentration_hhi": _herfindahl_index(buyer_amounts),
            }
        )

    tabular = pd.DataFrame(records).fillna(0.0)
    merged = tabular.merge(profiles, on="business_id", how="left")

    merged["revenue_to_txn_volume_ratio"] = merged["declared_monthly_revenue"] / (
        (merged["mean_amount"] * merged["n_transactions"]).replace(0, np.nan)
    )
    merged["revenue_to_txn_volume_ratio"] = merged["revenue_to_txn_volume_ratio"].fillna(0.0)

    ordered_cols = ["business_id", *TABULAR_FEATURE_COLUMNS]
    if "defaulted" in merged.columns:
        ordered_cols.append("defaulted")
    return merged[ordered_cols]