"""
Builds per-business temporal transaction graphs consumable by the T-GAT
embedder: a bipartite (business-side) graph of counterparty nodes with
edge features (amount, recency, on-time flag), bucketed into discrete
time windows so the model can attend over both structure and time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_COUNTERPARTIES = 32  # cap per business for fixed-size batching
N_TIME_WINDOWS = 6  # e.g. trailing 6 monthly windows


@dataclass
class BusinessGraph:
    business_id: str
    # [n_time_windows, max_counterparties, edge_feature_dim]
    edge_features: np.ndarray
    # [n_time_windows, max_counterparties] boolean mask, True = real counterparty present
    node_mask: np.ndarray


EDGE_FEATURE_DIM = 3  # normalized amount, recency weight, on_time flag


def build_business_graph(
    business_id: str,
    business_transactions: pd.DataFrame,
    reference_date: pd.Timestamp,
    max_counterparties: int = MAX_COUNTERPARTIES,
    n_time_windows: int = N_TIME_WINDOWS,
) -> BusinessGraph:
    tx = business_transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])

    window_days = 365 // n_time_windows
    tx["window_idx"] = (
        (reference_date - tx["transaction_date"]).dt.days // window_days
    ).clip(lower=0, upper=n_time_windows - 1)
    tx["window_idx"] = (n_time_windows - 1) - tx["window_idx"]  # 0 = oldest, last = most recent

    top_counterparties = (
        tx.groupby("counterparty_id")["amount"].sum().nlargest(max_counterparties).index.tolist()
    )
    counterparty_index = {cp: i for i, cp in enumerate(top_counterparties)}

    edge_features = np.zeros((n_time_windows, max_counterparties, EDGE_FEATURE_DIM), dtype=np.float32)
    node_mask = np.zeros((n_time_windows, max_counterparties), dtype=bool)

    max_amount = tx["amount"].max() if not tx.empty else 1.0
    max_amount = max_amount if max_amount > 0 else 1.0

    for _, row in tx.iterrows():
        cp = row["counterparty_id"]
        if cp not in counterparty_index:
            continue
        w = int(row["window_idx"])
        c = counterparty_index[cp]

        normalized_amount = float(row["amount"]) / max_amount
        recency_weight = (w + 1) / n_time_windows
        on_time_flag = float(bool(row.get("settled_on_time", True)))

        # accumulate (a counterparty can have multiple transactions per window)
        edge_features[w, c, 0] += normalized_amount
        edge_features[w, c, 1] = max(edge_features[w, c, 1], recency_weight)
        edge_features[w, c, 2] = (edge_features[w, c, 2] + on_time_flag) / 2 if node_mask[
            w, c
        ] else on_time_flag
        node_mask[w, c] = True

    return BusinessGraph(business_id=business_id, edge_features=edge_features, node_mask=node_mask)


def build_graph_batch(
    transactions: pd.DataFrame, business_ids: list[str], reference_date: pd.Timestamp
) -> list[BusinessGraph]:
    graphs = []
    grouped = {bid: g for bid, g in transactions.groupby("business_id")}
    for business_id in business_ids:
        biz_tx = grouped.get(business_id, transactions.iloc[0:0])
        graphs.append(build_business_graph(business_id, biz_tx, reference_date))
    return graphs