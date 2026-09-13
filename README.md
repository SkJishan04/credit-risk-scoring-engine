<div align="center">

# 💳 Algorithmic Credit Risk & Working Capital Scoring Engine

### A hybrid Graph Attention Network + Gradient Boosting system for scoring MSME & gig-economy creditworthiness from alternative transactional data

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-blue)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

<!--
  PLACEHOLDER — hero banner image
  Suggested: an abstract graph/network visual over a financial dashboard theme.
  Save as: docs/images/hero-banner.png  (recommended size: 1600x400)
-->
![Project Banner](./docs/images/hero-banner.png)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Motivation](#-motivation)
- [Key Features](#-key-features)
- [System Workflow](#-system-workflow)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Environment Variables](#-environment-variables)
- [API Usage & Examples](#-api-usage--examples)
- [Results & Screenshots](#-results--screenshots)
- [Evaluation Methodology](#-evaluation-methodology)
- [Engineering Notes](#-engineering-notes-real-bugs-found--fixed)
- [Testing](#-testing)
- [Docker](#-docker)
- [CI/CD](#-cicd)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [License](#license)

---

## 🔎 Overview

This project is a **production-shaped machine learning system** that scores the
creditworthiness of small businesses and gig-economy workers using **alternative
transactional data** — vendor payment histories, invoice timelines, and cash-flow
patterns — instead of traditional credit bureau history.

At its core is a **stacked ensemble**: a custom **Temporal Graph Attention Network
(T-GAT)** learns a dense representation of each business's transaction graph over
time, which is concatenated with hand-engineered financial features and fed into a
**calibrated XGBoost meta-model**. The result is a dynamic credit score, a
recommended interest rate, and a per-decision explanation — all served through a
typed, tested, containerized **FastAPI** service.

It's built to demonstrate the full lifecycle of an applied ML system: data
generation, feature engineering, graph representation learning, model stacking,
calibration, explainability, a real relational schema, a caching layer, automated
testing, CI/CD, and Docker packaging — not just a notebook with a `.predict()` call.

## 💡 Problem Statement

Banks and fintech lenders systematically under-serve **Micro, Small, and Medium
Enterprises (MSMEs)** and **gig-economy workers**, because traditional credit
bureaus have thin or no history on them. This forces lenders into one of two bad
outcomes:

- **Reject the applicant outright** → missed loan growth, financial exclusion.
- **Price risk using static, rule-based heuristics** → elevated Non-Performing
  Assets (NPAs) from mis-priced risk.

The transactional data *does* exist, though — payment timing, vendor
relationships, and cash-flow rhythm are rich, underused signals of financial
health. This project treats the **shape of a business's cash-flow graph over
time** as the primary risk signal, rather than a static credit history snapshot.

## 🎯 Motivation

> *"In corporate and SME banking, evaluating creditworthiness is bottlenecked by a
> lack of credit history. A hybrid ML pipeline using alternative transaction flow
> data can reduce false-positive default predictions and enable safer, automated
> working capital lending."*

This project was built to demonstrate applied competence across the full modern
AI/ML engineering stack: graph representation learning, model stacking and
calibration, explainability, production API design, database schema design,
caching resilience, automated testing, and containerized deployment — the kind of
end-to-end ownership expected of an AI/ML/GenAI engineer, not just model-fitting
in a notebook.

## ✨ Key Features

- 🕸️ **Custom Temporal Graph Attention Network (T-GAT)** — dependency-light, pure
  PyTorch implementation (no PyTorch Geometric) for maximum reproducibility.
- 🌲 **Stacked XGBoost meta-model** with isotonic probability calibration for
  trustworthy, well-calibrated default probabilities.
- 🔍 **Per-decision explainability** via SHAP — every score ships with the top
  contributing features.
- 📊 **Synthetic data generation** with a fully documented, interpretable
  generative process (payment jitter, vendor concentration, autocorrelated
  distress spirals) — disclosed transparently rather than hidden.
- ⚡ **Production-shaped FastAPI service** — typed schemas, dependency injection,
  structured logging, auto-generated OpenAPI docs.
- 🗄️ **Relational persistence** with PostgreSQL, SQLAlchemy 2.0, and
  Alembic-versioned migrations.
- 🚦 **Resilient caching** — Redis speeds up repeated requests, but a Redis outage
  degrades gracefully instead of failing scoring requests.
- 📈 **Experiment tracking** with MLflow (SQLite-backed, zero extra services).
- 🧪 **Real automated test suite** — unit, integration, and CI-enforced model
  quality gates (not vanity tests).
- 🐳 **Multi-stage Docker build** and **GitHub Actions CI/CD** pipeline.

## 🔄 System Workflow

```mermaid
flowchart TD
    A[📥 Client submits business profile + transactions] --> B{Cache hit?}
    B -- Yes --> C[⚡ Return cached score]
    B -- No / Redis unavailable --> D[🧮 Feature Engineering]
    D --> E[🕸️ Build temporal transaction graph]
    E --> F[🧠 T-GAT Embedding]
    D --> G[📐 Tabular Features]
    F --> H[🔗 Concatenate Features]
    G --> H
    H --> I[🌲 Calibrated XGBoost]
    I --> J[📊 Default Probability]
    J --> K[💯 Credit Score 300–900]
    J --> L[💵 Recommended Interest Rate]
    I --> M[🔍 SHAP Explanation]
    J --> N[🗄️ Persist to PostgreSQL]
    K --> O[📤 Response to Client]
    L --> O
    M --> O
```

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Vendor / Buyer Transactions] --> B[Feature Engineering<br/>cash-flow intervals, HHI, on-time %]
    A --> C[Temporal Graph Builder<br/>windowed bipartite graph]
    C --> D[T-GAT Embedder<br/>graph attention + GRU]
    B --> E[Feature Concatenation]
    D --> E
    E --> F[Calibrated XGBoost<br/>meta-model]
    F --> G[Default Probability]
    G --> H[Dynamic Credit Score 300-900]
    G --> I[Recommended Interest Rate]
    F --> J[SHAP Explainability]
```

<!--
  PLACEHOLDER — architecture illustration
  Suggested: a clean layered-diagram infographic (data → graph model → ensemble → API).
  Save as: docs/images/architecture-illustration.png
-->
![Architecture Illustration](./docs/images/architecture-illustration.png)

### Request lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Cache as Redis
    participant DB as PostgreSQL
    participant Model as T-GAT + XGBoost

    Client->>API: POST /api/v1/scoring
    API->>Cache: check cached response (non-fatal if Redis is down)
    alt cache hit
        Cache-->>API: cached score
    else cache miss
        API->>DB: upsert business + transactions
        API->>Model: features → embedding → probability
        Model-->>API: score, rate, SHAP factors
        API->>DB: persist score
        API->>Cache: store response (non-fatal if Redis is down)
    end
    API-->>Client: ScoreResponse
```

### Why a stacked ensemble, not a single model?

| Component | Role |
|---|---|
| **T-GAT** | Learns *relational + temporal* structure — who a business transacts with, how consistently, and how that evolves over time. |
| **Tabular features** | Captures interpretable, auditable signals (concentration ratios, on-time %) that a regulator or credit committee can sanity-check directly. |
| **XGBoost meta-model** | Combines both into a single, calibrated, well-understood tabular model — strong performance without sacrificing explainability. |

