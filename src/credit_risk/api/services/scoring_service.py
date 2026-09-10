"""Orchestrates feature engineering, model inference, and persistence for a scoring request."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from credit_risk.api.schemas import FeatureContribution, ScoreRequestSchema, ScoreResponseSchema
from credit_risk.db.repository import BusinessRepository
from credit_risk.explainability.shap_explainer import ScoreExplainer
from credit_risk.features.engineering import build_tabular_features
from credit_risk.features.graph_builder import build_graph_batch
from credit_risk.models.ensemble import CreditRiskEnsemble


class ScoringService:
    def __init__(self, ensemble: CreditRiskEnsemble, db: Session):
        self.ensemble = ensemble
        self.db = db
        self.explainer = ScoreExplainer(ensemble.xgb_scorer)

    def score(self, request: ScoreRequestSchema) -> ScoreResponseSchema:
        repo = BusinessRepository(self.db)
        repo.upsert_business(request.business_profile)
        repo.replace_transactions(request.business_profile.business_id, request.transactions)

        transactions_df = pd.DataFrame([tx.model_dump() for tx in request.transactions])
        profiles_df = pd.DataFrame([request.business_profile.model_dump()])

        tabular = build_tabular_features(transactions_df, profiles_df)
        reference_date = pd.Timestamp(transactions_df["transaction_date"].max())
        graphs = build_graph_batch(
            transactions_df, tabular["business_id"].tolist(), reference_date
        )

        X = self.ensemble.build_feature_matrix(tabular, graphs)
        results = self.ensemble.score(tabular, graphs)
        result = results[0]

        top_factors = self.explainer.top_contributing_factors(X[0])

        repo.save_score(result)

        return ScoreResponseSchema(
            business_id=result.business_id,
            default_probability=round(result.default_probability, 4),
            credit_score=result.credit_score,
            recommended_interest_rate_pct=result.recommended_interest_rate_pct,
            top_contributing_factors=[FeatureContribution(**f) for f in top_factors],
            cached=False,
        )