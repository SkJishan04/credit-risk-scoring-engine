"""SHAP-based explanations for individual scoring decisions."""

from __future__ import annotations

import numpy as np
import shap

from credit_risk.models.xgboost_scorer import XGBoostScorer


class ScoreExplainer:
    def __init__(self, scorer: XGBoostScorer):
        if scorer.booster is None:
            raise RuntimeError("Scorer must be fit/loaded before explaining.")
        self.scorer = scorer
        self._explainer = shap.TreeExplainer(scorer.booster)

    def top_contributing_factors(self, X_row: np.ndarray, top_k: int = 5) -> list[dict]:
        """Returns the top_k features pushing this single prediction toward
        (positive value) or away from (negative value) default."""
        shap_values = self._explainer.shap_values(X_row.reshape(1, -1))[0]
        feature_names = self.scorer.feature_names or [f"f_{i}" for i in range(len(shap_values))]

        contributions = sorted(
            zip(feature_names, shap_values, strict=True), key=lambda kv: abs(kv[1]), reverse=True
        )[:top_k]

        return [
            {"feature": name, "shap_value": round(float(value), 4)}
            for name, value in contributions
        ]