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

## 🛠️ Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **API** | FastAPI + Pydantic v2 | Async, fully typed, auto-generated OpenAPI docs |
| **Graph / Deep Learning** | PyTorch (custom T-GAT, no PyG) | Avoids CUDA/PyG wheel fragility — trivially reproducible |
| **Tabular Model** | XGBoost + isotonic calibration | Strong tabular baseline with well-calibrated probabilities |
| **Explainability** | SHAP | Per-decision feature attribution, essential in a regulated domain |
| **Database** | PostgreSQL + SQLAlchemy 2.0 + Alembic | Relational integrity with versioned, reviewable schema migrations |
| **Cache** | Redis (non-critical path) | Speeds up repeat requests; scoring degrades gracefully if Redis is down |
| **Experiment Tracking** | MLflow (SQLite-backed) | Reproducible run history, zero extra infrastructure |
| **Testing** | Pytest + fakeredis + httpx | Unit, integration, and model-quality gate tests |
| **CI/CD** | GitHub Actions | Lint → type-check → test → Docker build on every push |
| **Packaging** | Docker (multi-stage) | Small, non-root, production-conscious image |
| **Code Quality** | Ruff + Mypy | Fast linting and static typing across the codebase |

## 📁 Project Structure

```text
credit-risk-scoring-engine/
├── src/credit_risk/
│   ├── config.py                # Centralized, environment-driven settings
│   ├── logging_config.py        # Structured logging (structlog)
│   ├── data/
│   │   ├── schemas.py           # Core Pydantic data contracts
│   │   └── synthetic_generator.py  # Documented synthetic MSME data generator
│   ├── features/
│   │   ├── engineering.py       # Tabular feature engineering
│   │   └── graph_builder.py     # Temporal bipartite graph construction
│   ├── models/
│   │   ├── tgat.py              # Temporal Graph Attention Network
│   │   ├── xgboost_scorer.py    # Calibrated XGBoost meta-model
│   │   └── ensemble.py          # Stacked ensemble + scoring transformation
│   ├── training/
│   │   ├── train_pipeline.py    # End-to-end training pipeline
│   │   └── evaluate.py          # Evaluation metrics
│   ├── explainability/
│   │   └── shap_explainer.py    # SHAP-based per-decision explanations
│   ├── db/
│   │   ├── models.py            # SQLAlchemy ORM models
│   │   ├── session.py           # DB session management
│   │   └── repository.py        # Data-access layer
│   └── api/
│       ├── main.py              # FastAPI application entrypoint
│       ├── schemas.py           # API request/response schemas
│       ├── dependencies.py      # Dependency injection (DB, cache, model)
│       ├── services/            # Business logic (cache, scoring orchestration)
│       └── routers/             # Route handlers (health, scoring)
├── alembic/                     # Versioned database migrations
├── scripts/                     # CLI entrypoints (generate_data.py, train.py)
├── tests/
│   ├── unit/                    # Feature engineering, graph builder, XGBoost wrapper
│   ├── integration/             # Full API flow tests
│   └── evaluation/              # CI-enforced model-quality gate
├── .github/workflows/ci.yml     # GitHub Actions pipeline
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # API + Postgres + Redis stack
└── README.md
```

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 16 and Redis 7 (locally installed, or via Docker)
- (Optional) Docker & Docker Compose

### Local setup (no Docker)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/credit-risk-scoring-engine.git
cd credit-risk-scoring-engine

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env

# 5. Start Postgres + Redis (however you prefer)
docker compose up -d db redis
# ...or install natively: brew install postgresql@16 redis / apt-get install postgresql redis-server

# 6. Apply database migrations
alembic upgrade head

# 7. Train the model (writes ./artifacts)
python scripts/train.py

# 8. Run the API
uvicorn credit_risk.api.main:app --reload
```

API docs available at: **`http://localhost:8000/docs`**

### Quick setup with Docker

```bash
cp .env.example .env
python scripts/train.py         # produces ./artifacts, mounted read-only into the container
docker compose up --build
```

## 🔑 Environment Variables

All configuration is environment-driven (`pydantic-settings`). See `.env.example`
for the full reference — summarized below:

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg://credit_user:credit_pass@localhost:5432/credit_risk_db` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `SCORE_CACHE_TTL_SECONDS` | Cache TTL for scoring responses | `300` |
| `MODEL_ARTIFACT_DIR` | Directory for trained model artifacts | `./artifacts` |
| `MLFLOW_TRACKING_URI` | MLflow tracking store | `sqlite:///mlflow.db` |
| `RANDOM_SEED` | Global reproducibility seed | `42` |
| `SYNTHETIC_N_BUSINESSES` | Synthetic dataset size for training | `4000` |

## 🔌 API Usage & Examples

### Score a business

```bash
curl -X POST http://localhost:8000/api/v1/scoring \
  -H "Content-Type: application/json" \
  -d '{
        "business_profile": {
          "business_id": "biz_00042",
          "sector": "retail",
          "months_active": 24,
          "declared_monthly_revenue": 50000,
          "n_active_vendors": 3,
          "n_active_buyers": 5
        },
        "transactions": [
          {
            "transaction_id": "tx_0",
            "business_id": "biz_00042",
            "counterparty_id": "vendor_0",
            "amount": 1200.0,
            "direction": "outflow",
            "transaction_date": "2024-01-01",
            "invoice_due_date": "2024-01-15",
            "settled_on_time": true
          }
        ]
      }'
```

### Response

```json
{
  "business_id": "biz_00042",
  "default_probability": 0.3056,
  "credit_score": 717,
  "recommended_interest_rate_pct": 14.72,
  "top_contributing_factors": [
    { "feature": "pct_settled_on_time", "shap_value": -0.3089 },
    { "feature": "revenue_to_txn_volume_ratio", "shap_value": 0.0640 },
    { "feature": "n_transactions", "shap_value": 0.0193 }
  ],
  "cached": false
}
```

> 💡 `transaction_id` only needs to be unique **within** a given `business_id`,
> not globally across the platform — see [Engineering Notes](#-engineering-notes-real-bugs-found--fixed).

## 📸 Results & Screenshots

<!--
  PLACEHOLDER — Swagger UI / API docs screenshot
  Save as: docs/images/swagger-ui.png
-->
![Swagger UI](./docs/images/swagger-ui.png)

<!--
  PLACEHOLDER — MLflow experiment comparison screenshot
  Save as: docs/images/mlflow-experiments.png
-->
![MLflow Experiment Tracking](./docs/images/mlflow-experiments.png)

### Sample training run

```text
2026-09-11 06:34:45 | evaluation_metrics
    auc_roc=0.776   auc_pr=0.7504   brier_score=0.1901   max_f1=0.7627
2026-09-11 06:34:45 | training_pipeline_complete
    tgat_path=artifacts/tgat_embedder.pt   xgb_path=artifacts/xgb_scorer.json
```

| Metric | Value | What it means |
|---|---|---|
| **AUC-ROC** | 0.776 | Strong discrimination between defaulting and non-defaulting businesses |
| **AUC-PR** | 0.750 | Solid precision-recall trade-off, useful under class imbalance |
| **Brier Score** | 0.190 | Well-calibrated probabilities — meaningful for interest-rate pricing |
| **Max F1** | 0.763 | Healthy operating-point performance |

> These metrics are on **synthetic data with a documented, interpretable
> generative process** — see [Limitations](#-limitations) for what they do and
> don't claim.

## 📐 Evaluation Methodology

- **Train/test split** is stratified on the default label and held **constant**
  across both the T-GAT pretraining phase and the XGBoost fit — no test-set
  leakage occurs between stages.
- **Metrics reported**: AUC-ROC and AUC-PR (discrimination), Brier score
  (calibration), and max-F1 (operating-point summary).
- **Calibration** is handled by a standalone `IsotonicRegression` fit on the base
  model's held-out validation-fold probabilities — not
  `CalibratedClassifierCV(cv="prefit")`, which was removed in scikit-learn ≥1.6.
  This keeps calibration correct (no re-fitting on validation data) and
  independent of a specific sklearn version.
- **Every run is logged to MLflow** (params, metrics, artifacts) for
  longitudinal comparison across experiments.
- A **CI-enforced model-quality gate** (`tests/evaluation/`) fails the build if
  the pipeline ever regresses below a minimum AUC-ROC / Brier bar — the model's
  correctness is a tested contract, not just a training-script output.

## 🐛 Engineering Notes: Real Bugs Found & Fixed

This project went through a genuine debugging pass against **live PostgreSQL and
Redis instances**, not just isolated unit tests. Two production-relevant issues
surfaced and were fixed properly:

### 1. Transaction IDs were a global primary key, not scoped per business

The original schema made `transaction_id` the sole primary key on the
`transactions` table — silently assuming transaction references are unique
**across every business on the platform**. In reality, a client's transaction ID
(e.g. their own invoice number) is only guaranteed unique **within their own
books**. Two different businesses legitimately sending `"tx_0"`, `"tx_1"`, ...
would collide on insert.

**Fix:** moved to a surrogate UUID primary key, with uniqueness enforced on the
composite `(business_id, transaction_id)` instead, via a proper Alembic
migration (`0002_transaction_surrogate_key.py`) — not a workaround.

### 2. A Redis outage could take down the entire scoring endpoint

Cache read/write calls originally sat **outside** the endpoint's error handling.
A momentarily unreachable Redis instance crashed the whole scoring request with
an unhandled 500, even though caching is purely a performance optimization.

**Fix:** wrapped both cache operations to catch `redis.RedisError`, log a
warning, and fall through to computing (and simply not caching) the score —
verified by killing Redis mid-session and confirming `200 OK` responses continue.

```mermaid
flowchart LR
    A[🔴 Bug: Redis down] --> B[Old: unhandled 500]
    A --> C[✅ New: warning logged]
    C --> D[Score computed normally]
    D --> E[200 OK, cached: false]
```

> Both fixes are covered by the automated test suite and documented as a case
> study in resilient system design, not just "getting it to run."

## 🧪 Testing

```bash
make test
# or
pytest -q --cov=credit_risk --cov-report=term-missing
```

| Suite | What it covers |
|---|---|
| `tests/unit/` | Feature engineering, temporal graph construction, XGBoost wrapper |
| `tests/integration/` | Full API request → response flow against a freshly trained model, with a fake Redis backend |
| `tests/evaluation/` | CI-enforced minimum AUC-ROC / Brier-score gate on a held-out split |

```text
16 passed in 16.94s
```

> ⚠️ The integration suite currently shares `DATABASE_URL` with the local dev
> environment rather than an isolated test database — acceptable at this
> project's scope, documented under [Limitations](#-limitations), and listed
> as a concrete [future improvement](#-future-improvements).

## 🐳 Docker

Multi-stage build — a `builder` stage compiles wheels, and a slim `runtime` stage
runs as a **non-root user** with a built-in health check.

```bash
cp .env.example .env
python scripts/train.py          # produces ./artifacts locally first
docker compose up --build
```

```mermaid
flowchart LR
    A[docker compose up] --> B[db: PostgreSQL 16]
    A --> C[redis: Redis 7]
    A --> D[api: FastAPI service]
    D -- depends_on: healthy --> B
    D -- depends_on: healthy --> C
    D -- mounts read-only --> E[./artifacts]
```

Services:

| Service | Image | Purpose |
|---|---|---|
| `db` | `postgres:16-alpine` | Primary relational store |
| `redis` | `redis:7-alpine` | Response cache |
| `api` | Built from `Dockerfile` | FastAPI scoring service |

## 🔁 CI/CD

Every push and pull request to `main` runs a full **GitHub Actions** pipeline:

```mermaid
flowchart LR
    A[git push] --> B[Lint — ruff]
    B --> C[Type Check — mypy]
    C --> D[Test — pytest + coverage]
    D --> E[Upload coverage report]
    E --> F[Docker image build]
```

- **Lint & type-check** run against the full `src/` and `tests/` trees.
- **Tests** run against real, service-container Postgres and Redis instances
  (not mocks) — the same rigor as local development.
- **Docker build** validates the production image builds cleanly on every push.

