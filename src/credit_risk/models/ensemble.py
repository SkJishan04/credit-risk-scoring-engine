"""
Stacked ensemble tying the T-GAT embedder and the XGBoost meta-model
together, plus the business-facing scoring transformation (default
probability -> credit score -> recommended interest rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from credit_risk.features.engineering import TABULAR_FEATURE_COLUMNS
from credit_risk.features.graph_builder import BusinessGraph
from credit_risk.models.tgat import TGATEmbedder
from credit_risk.models.xgboost_scorer import XGBoostScorer

EMBEDDING_DIM = 16
BASE_RATE_PCT = 8.0  # risk-free-ish floor for a well-scored borrower
MAX_RISK_PREMIUM_PCT = 22.0  # additional premium at p_default -> 1.0


@dataclass
class ScoringResult:
    business_id: str
    default_probability: float
    credit_score: int  # 300-900 scale, higher = safer
    recommended_interest_rate_pct: float


def graphs_to_tensors(graphs: list[BusinessGraph]) -> tuple[torch.Tensor, torch.Tensor]:
    edge_features = torch.tensor(np.stack([g.edge_features for g in graphs]), dtype=torch.float32)
    node_mask = torch.tensor(np.stack([g.node_mask for g in graphs]), dtype=torch.bool)
    return edge_features, node_mask


def default_probability_to_score(p_default: float) -> int:
    """Map [0, 1] default probability to a 300-900 score (higher = safer)."""
    p_default = min(max(p_default, 1e-4), 1 - 1e-4)
    score = 900 - (p_default * 600)
    return int(round(score))


def score_to_interest_rate(p_default: float) -> float:
    rate = BASE_RATE_PCT + p_default * MAX_RISK_PREMIUM_PCT
    return round(rate, 2)


class CreditRiskEnsemble:
    """Loads trained artifacts and produces end-to-end scoring results."""

    def __init__(self, tgat: TGATEmbedder, xgb_scorer: XGBoostScorer):
        self.tgat = tgat
        self.xgb_scorer = xgb_scorer

    @classmethod
    def load(cls, tgat_path: Path, xgb_path: Path, random_seed: int = 42) -> "CreditRiskEnsemble":
        tgat = TGATEmbedder(embedding_dim=EMBEDDING_DIM)
        tgat.load_state_dict(torch.load(tgat_path, map_location="cpu"))
        tgat.eval()
        xgb_scorer = XGBoostScorer.load(xgb_path, random_seed=random_seed)
        return cls(tgat=tgat, xgb_scorer=xgb_scorer)

    def build_feature_matrix(
        self, tabular_df, graphs: list[BusinessGraph]
    ) -> np.ndarray:
        edge_features, node_mask = graphs_to_tensors(graphs)
        embeddings = self.tgat.embed(edge_features, node_mask)  # [N, embedding_dim]
        tabular = tabular_df[TABULAR_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        return np.concatenate([tabular, embeddings], axis=1)

    def score(self, tabular_df, graphs: list[BusinessGraph]) -> list[ScoringResult]:
        X = self.build_feature_matrix(tabular_df, graphs)
        probabilities = self.xgb_scorer.predict_proba(X)

        results = []
        for business_id, p_default in zip(tabular_df["business_id"], probabilities, strict=True):
            results.append(
                ScoringResult(
                    business_id=business_id,
                    default_probability=float(p_default),
                    credit_score=default_probability_to_score(float(p_default)),
                    recommended_interest_rate_pct=score_to_interest_rate(float(p_default)),
                )
            )
        return results


def feature_names_with_embedding() -> list[str]:
    return [*TABULAR_FEATURE_COLUMNS, *[f"tgat_emb_{i}" for i in range(EMBEDDING_DIM)]]