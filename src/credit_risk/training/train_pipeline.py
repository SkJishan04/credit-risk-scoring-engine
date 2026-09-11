"""
End-to-end training pipeline:
  1. Generate/load synthetic data
  2. Build tabular features + per-business transaction graphs
  3. Pretrain T-GAT embedder with the auxiliary default-prediction head
  4. Freeze T-GAT, extract embeddings, train calibrated XGBoost meta-model
  5. Log parameters/metrics/artifacts to MLflow
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim

from credit_risk.config import Settings, get_settings
from credit_risk.data.synthetic_generator import generate_synthetic_dataset
from credit_risk.features.engineering import build_tabular_features
from credit_risk.features.graph_builder import build_graph_batch
from credit_risk.logging_config import get_logger
from credit_risk.models.ensemble import (
    EMBEDDING_DIM,
    feature_names_with_embedding,
    graphs_to_tensors,
)
from credit_risk.models.tgat import TGATEmbedder
from credit_risk.models.xgboost_scorer import XGBoostScorer
from credit_risk.training.evaluate import evaluate_binary_classifier

logger = get_logger(__name__)


def pretrain_tgat(
    edge_features: torch.Tensor,
    node_mask: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 42,
) -> TGATEmbedder:
    torch.manual_seed(seed)
    model = TGATEmbedder(embedding_dim=EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, logits = model(edge_features, node_mask)
        loss = criterion(logits.squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("tgat_pretrain_epoch", epoch=epoch + 1, loss=float(loss.item()))

    model.eval()
    return model


def run_training_pipeline(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="tgat_xgboost_stack"):
        mlflow.log_params(
            {
                "n_businesses": settings.synthetic_n_businesses,
                "random_seed": settings.random_seed,
                "embedding_dim": EMBEDDING_DIM,
            }
        )

        logger.info("generating_synthetic_dataset")
        dataset = generate_synthetic_dataset(
            n_businesses=settings.synthetic_n_businesses, seed=settings.random_seed
        )

        logger.info("building_tabular_features")
        tabular = build_tabular_features(dataset.transactions, dataset.profiles)

        logger.info("building_transaction_graphs")
        reference_date = pd.Timestamp(dataset.transactions["transaction_date"].max())
        graphs = build_graph_batch(
            dataset.transactions, tabular["business_id"].tolist(), reference_date
        )

        edge_features, node_mask = graphs_to_tensors(graphs)
        labels = torch.tensor(tabular["defaulted"].astype(float).to_numpy(), dtype=torch.float32)

        # Split indices once, reuse consistently for both T-GAT pretraining and XGBoost.
        idx = np.arange(len(tabular))
        idx_train, idx_test = train_test_split(
            idx, test_size=0.2, random_state=settings.random_seed, stratify=labels.numpy()
        )

        logger.info("pretraining_tgat_embedder")
        tgat = pretrain_tgat(
            edge_features[idx_train],
            node_mask[idx_train],
            labels[idx_train],
            seed=settings.random_seed,
        )

        with torch.no_grad():
            all_embeddings = tgat.embed(edge_features, node_mask)

        from credit_risk.features.engineering import TABULAR_FEATURE_COLUMNS

        X_tabular = tabular[TABULAR_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        X_full = np.concatenate([X_tabular, all_embeddings], axis=1)
        y = tabular["defaulted"].astype(int).to_numpy()

        logger.info("training_xgboost_scorer")
        scorer = XGBoostScorer(random_seed=settings.random_seed)
        fit_info = scorer.fit(
            X_full[idx_train], y[idx_train], feature_names=feature_names_with_embedding()
        )

        logger.info("evaluating_model")
        y_proba_test = scorer.predict_proba(X_full[idx_test])
        metrics = evaluate_binary_classifier(y[idx_test], y_proba_test)
        logger.info("evaluation_metrics", **metrics)
        mlflow.log_metrics(metrics)
        mlflow.log_params(fit_info)

        settings.model_artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tgat.state_dict(), settings.tgat_model_path)
        scorer.save(settings.xgb_model_path)
        mlflow.log_artifact(str(settings.tgat_model_path))
        mlflow.log_artifact(str(settings.xgb_model_path))

        logger.info(
            "training_pipeline_complete",
            tgat_path=str(settings.tgat_model_path),
            xgb_path=str(settings.xgb_model_path),
        )
        return metrics


if __name__ == "__main__":
    run_training_pipeline()