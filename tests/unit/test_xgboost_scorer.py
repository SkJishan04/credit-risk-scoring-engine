"""Unit tests for the XGBoost meta-model wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.models.xgboost_scorer import XGBoostScorer


@pytest.fixture
def synthetic_classification_data():
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    # y depends on a clear linear signal so the model can actually learn it
    logits = X[:, 0] * 2.0 - X[:, 1] * 1.5
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X.astype(np.float32), y


def test_fit_and_predict_proba_in_range(synthetic_classification_data):
    X, y = synthetic_classification_data
    scorer = XGBoostScorer(random_seed=0)
    scorer.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])

    proba = scorer.predict_proba(X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_before_fit_raises():
    scorer = XGBoostScorer()
    with pytest.raises(RuntimeError):
        scorer.predict_proba(np.zeros((1, 3)))


def test_learns_better_than_random(synthetic_classification_data):
    from sklearn.metrics import roc_auc_score

    X, y = synthetic_classification_data
    scorer = XGBoostScorer(random_seed=0)
    scorer.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])
    proba = scorer.predict_proba(X)
    assert roc_auc_score(y, proba) > 0.7


def test_save_and_load_roundtrip(tmp_path, synthetic_classification_data):
    X, y = synthetic_classification_data
    scorer = XGBoostScorer(random_seed=0)
    scorer.fit(X, y, feature_names=[f"f{i}" for i in range(X.shape[1])])

    path = tmp_path / "model.joblib"
    scorer.save(path)

    loaded = XGBoostScorer.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), scorer.predict_proba(X))