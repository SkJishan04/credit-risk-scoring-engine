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