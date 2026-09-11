"""XGBoost meta-model: tabular features + T-GAT embedding -> default probability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split


class XGBoostScorer:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.booster: xgb.XGBClassifier | None = None
        self.calibrator: IsotonicRegression | None = None
        self.feature_names: list[str] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
        self.feature_names = feature_names

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )

        base_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=self.random_seed,
            early_stopping_rounds=20,
        )
        base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.booster = base_model

        # Calibrate probabilities on the held-out fold so recommended interest
        # rates map to genuinely well-calibrated default probabilities.
        #
        # We fit a standalone IsotonicRegression on the base model's raw
        # validation-set probabilities rather than sklearn's
        # CalibratedClassifierCV(cv="prefit"): that API was removed in
        # scikit-learn >= 1.6, and this approach is simpler, has no
        # sklearn-version coupling, and calibrates the *already-fit* model
        # (no data leakage from re-fitting on the validation fold).
        raw_val_proba = base_model.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(raw_val_proba, y_val)

        return {
            "n_train": len(X_train),
            "n_val": len(X_val),
            "best_iteration": base_model.best_iteration,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None or self.calibrator is None:
            raise RuntimeError("Model has not been fit or loaded yet.")
        raw_proba = self.booster.predict_proba(X)[:, 1]
        return self.calibrator.predict(raw_proba)

    def save(self, path: Path) -> None:
        if self.booster is None or self.calibrator is None:
            raise RuntimeError("Cannot save an unfit model.")
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(
            {
                "booster": self.booster,
                "calibrator": self.calibrator,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, random_seed: int = 42) -> XGBoostScorer:
        import joblib

        payload = joblib.load(path)
        instance = cls(random_seed=random_seed)
        instance.booster = payload["booster"]
        instance.calibrator = payload["calibrator"]
        instance.feature_names = payload["feature_names"]
        return instance