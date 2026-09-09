"""Model evaluation metrics for the default-prediction task."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate_binary_classifier(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Returns discrimination (AUC-ROC, AUC-PR), calibration (Brier score), and
    an operating-point metric (max F1 across thresholds) — the metric set a
    credit-risk reviewer would actually ask for, not just accuracy.
    """
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    max_f1 = float(np.max(f1_scores))

    return {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "brier_score": round(float(brier), 4),
        "max_f1": round(max_f1, 4),
    }