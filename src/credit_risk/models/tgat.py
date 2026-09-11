"""
Temporal Graph Attention Network (T-GAT) embedder.

Dependency-light, pure PyTorch implementation (no PyTorch Geometric) so the
project stays trivially reproducible. For each business, at each of the
N trailing time windows we run additive attention over that window's
counterparty edge-feature set to produce a window embedding, then roll the
per-window embeddings through a GRU to capture temporal evolution. The final
GRU hidden state is the business's dense representation, used both as an
auxiliary default-probability head (for supervised pretraining) and as extra
input features to the downstream XGBoost meta-model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GraphAttentionWindow(nn.Module):
    """Additive attention over counterparties within a single time window."""

    def __init__(self, edge_feature_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, edge_features: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        edge_features: [batch, max_counterparties, edge_feature_dim]
        node_mask:     [batch, max_counterparties] (bool)
        returns:       [batch, hidden_dim]
        """
        h = torch.tanh(self.edge_proj(edge_features))  # [B, C, H]
        scores = self.attn_score(h).squeeze(-1)  # [B, C]
        scores = scores.masked_fill(~node_mask, float("-inf"))

        # windows with zero real counterparties -> uniform (masked) softmax would NaN;
        # guard by forcing an all-false mask to zero attention weights.
        any_real = node_mask.any(dim=-1, keepdim=True)
        scores = torch.where(any_real, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = weights * node_mask.float()

        pooled = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # [B, H]
        return pooled


class TGATEmbedder(nn.Module):
    """Sequence of per-window graph attention followed by a GRU over time."""

    def __init__(
        self,
        edge_feature_dim: int = 3,
        hidden_dim: int = 32,
        gru_hidden_dim: int = 32,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.window_attn = GraphAttentionWindow(edge_feature_dim, hidden_dim)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim, hidden_size=gru_hidden_dim, batch_first=True
        )
        self.embedding_head = nn.Linear(gru_hidden_dim, embedding_dim)
        self.default_head = nn.Linear(embedding_dim, 1)  # auxiliary supervised head

    def forward(
        self, edge_features: torch.Tensor, node_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        edge_features: [batch, n_windows, max_counterparties, edge_feature_dim]
        node_mask:     [batch, n_windows, max_counterparties]
        returns:       embedding [batch, embedding_dim], default_logit [batch, 1]
        """
        batch, n_windows, max_cp, feat_dim = edge_features.shape
        flat_edges = edge_features.reshape(batch * n_windows, max_cp, feat_dim)
        flat_mask = node_mask.reshape(batch * n_windows, max_cp)

        window_embeds = self.window_attn(flat_edges, flat_mask)  # [B*W, H]
        window_embeds = window_embeds.reshape(batch, n_windows, -1)

        _, final_hidden = self.temporal_gru(window_embeds)  # [1, B, gru_hidden]
        final_hidden = final_hidden.squeeze(0)

        embedding = self.embedding_head(final_hidden)
        default_logit = self.default_head(embedding)
        return embedding, default_logit

    @torch.no_grad()
    def embed(self, edge_features: torch.Tensor, node_mask: torch.Tensor) -> np.ndarray:
        self.eval()
        embedding, _ = self.forward(edge_features, node_mask)
        return embedding.cpu().numpy()