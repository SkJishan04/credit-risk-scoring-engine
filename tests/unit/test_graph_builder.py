"""Unit tests for the temporal graph construction module."""

from __future__ import annotations

import pandas as pd

from credit_risk.features.graph_builder import (
    EDGE_FEATURE_DIM,
    MAX_COUNTERPARTIES,
    N_TIME_WINDOWS,
    build_business_graph,
)


def test_graph_shape_matches_constants(sample_transactions_df):
    business_id = sample_transactions_df["business_id"].iloc[0]
    biz_tx = sample_transactions_df[sample_transactions_df["business_id"] == business_id]
    reference_date = pd.Timestamp(sample_transactions_df["transaction_date"].max())

    graph = build_business_graph(business_id, biz_tx, reference_date)

    assert graph.edge_features.shape == (N_TIME_WINDOWS, MAX_COUNTERPARTIES, EDGE_FEATURE_DIM)
    assert graph.node_mask.shape == (N_TIME_WINDOWS, MAX_COUNTERPARTIES)


def test_graph_mask_matches_nonzero_edges(sample_transactions_df):
    business_id = sample_transactions_df["business_id"].iloc[0]
    biz_tx = sample_transactions_df[sample_transactions_df["business_id"] == business_id]
    reference_date = pd.Timestamp(sample_transactions_df["transaction_date"].max())

    graph = build_business_graph(business_id, biz_tx, reference_date)

    masked_out_positions = ~graph.node_mask
    assert (graph.edge_features[masked_out_positions] == 0).all()


def test_empty_transactions_produce_all_false_mask():
    empty_df = pd.DataFrame(
        columns=["counterparty_id", "amount", "direction", "transaction_date", "settled_on_time"]
    )
    reference_date = pd.Timestamp("2024-01-01")
    graph = build_business_graph("biz_empty", empty_df, reference_date)
    assert not graph.node_mask.any()