# KKBox Churn Prediction — End-to-End MLOps Pipeline

> **English** (primary) · [Español](#versión-en-español) (secondary)

A production-grade machine learning system built on the [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) dataset. The project goes beyond model training — it implements a complete local MLOps stack with automated retraining, drift monitoring, CI/CD pipelines, and a REST API ready for deployment.

---

## What This Demonstrates

| Skill Area | Implementation |
| --- | --- |
| **ML Engineering** | LightGBM + XGBoost + CatBoost ensemble · Optuna tuning · temporal validation split |
| **Feature Engineering** | 48 features across transactions, membership, listening behavior, and expiry signals |
| **MLOps** | MLflow experiment tracking + model registry · automated quality gates · model promotion |
| **Software Engineering** | FastAPI service · Pydantic contracts · async SQLAlchemy · 29-test suite (unit + integration) |
| **Infrastructure** | Docker multi-stage build · Docker Compose (5 services) · PostgreSQL feature store |
| **CI/CD** | Jenkins pipelines: CI on every commit, CD on merge, monthly retrain, daily monitoring |
| **Monitoring** | Evidently AI data drift · performance tracking against ground truth · Grafana dashboards |

---

## Model Results

Best public score: **0.23412 log loss** (top ~30% of competition leaderboard at close).

| Submission | Public Score | Private Score | Description |
| --- | --- | --- | --- |
| v1 — Tuned LightGBM | 0.37856 | — | Baseline, 30 features, random split (overfit) |
| v3 — Temporal fix | 0.30398 | — | Fixed `days_since_last` date-offset bug |
| v4 — Expiry features | 0.23528 | — | +6 membership expiry signals |
| v5 — Best single model | 0.23504 | 0.23494 | 48 features, honest temporal split |
| v8 — CatBoost | 0.23986 | 0.23963 | Single CatBoost, 48 features |
| v9 — LGBM + XGB blend | 0.23975 | 0.23945 | 2-model blend |
| **v10 — 3-model blend** | **0.23436** | **0.23412** | LightGBM + XGBoost + CatBoost |

Key insight: `days_until_expire` is the strongest single predictor — an expired membership with `cancel_at_expire=1` is almost certain churn.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                     Docker Compose                       │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ FastAPI  │   │  MLflow  │   │ Jenkins  │            │
│  │  :8000   │   │  :5000   │   │  :8080   │            │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘            │
│       │              │              │                   │
│  ┌────▼──────────────▼──────────────▼─────┐            │
│  │              PostgreSQL :5432           │            │
│  │  features_monthly · predictions ·      │            │
│  │  model_versions · ground_truth ·       │            │
│  │  drift_reports                         │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
│  ┌──────────┐                                           │
│  │ Grafana  │  ← drift score · churn rate · latency    │
│  │  :3000   │                                           │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

**Prediction flow:**

1. Monthly batch pipeline computes 48 features from raw data → stores in `features_monthly`
2. FastAPI `/predict` and `/predict/batch` serve real-time predictions (model loaded once at startup)
3. Jenkins retrain pipeline runs on the 1st of each month: compute → train → quality gate → promote
4. Daily monitoring pipeline checks data drift and model performance → alerts via GitHub Issues

---

## Tech Stack

```text
ML Training      LightGBM · XGBoost · CatBoost · scikit-learn · Optuna
Experiment Track MLflow (tracking server + model registry + artifact store)
API              FastAPI · Pydantic v2 · async SQLAlchemy · httpx
Database         PostgreSQL 15
Containers       Docker (multi-stage) · Docker Compose
CI/CD            Jenkins (4 pipelines: CI, CD, retrain, monitoring)
Monitoring       Evidently AI · Grafana 10.4 · psycopg2
Testing          pytest · pytest-asyncio · 29 tests (unit + integration)
```

---

## Quick Start

```bash
# 1. Clone and configure
git clone <repo-url> && cd Churn_project
cp .env.example .env          # edit secrets if needed

# 2. Start all services
docker compose up --build -d

# 3. Verify
curl http://localhost:8000/health
```

| Service | URL | Credentials |
| --- | --- | --- |
| REST API + Swagger | <http://localhost:8000/docs> | `X-API-Key: changeme` |
| MLflow UI | <http://localhost:5000> | — |
| Jenkins | <http://localhost:8080> | admin / admin |
| Grafana | <http://localhost:3000> | admin / admin |
| PostgreSQL | localhost:5432 | see `.env` |

---

## Makefile Commands

```bash
make up          # Start full stack (docker compose up --build -d)
make down        # Stop stack
make test        # Run pytest inside the API container
make retrain     # Full retrain cycle: compute features → train → register
make predict     # Monthly batch prediction for current period
make drift       # Generate Evidently drift report
make promote     # Promote Staging model to Production in MLflow
make logs        # Tail API logs
make pipeline    # End-to-end: compute-features + retrain + predict
```

---

## Project Structure

```text
src/
  api/          FastAPI service — schemas, config, routes, dependencies
  models/       Training scripts (numbered 06–17, each a new experiment)
  pipeline/     Batch feature computation + batch prediction (CLI)
  monitoring/   Drift detection, performance tracking, weekly HTML reports
infra/
  docker/       Dockerfiles (API multi-stage, MLflow, Jenkins)
  postgres/     Schema SQL — 5 production tables
  jenkins/      Jenkinsfile.ci · Jenkinsfile.cd · Jenkinsfile.retrain · Jenkinsfile.monitoring
  grafana/      Dashboard JSON + provisioning config
tests/
  unit/         Feature engineering + Pydantic validation (16 tests)
  integration/  API endpoints — auth, predict, batch, health (13 tests)
notebooks/      EDA and model experiments
submissions/    Kaggle CSV history with scores
reports/
  monitoring/   Evidently HTML drift reports (generated)
```

---

## MLOps Phases

| Phase | Status | Description |
| --- | --- | --- |
| 0 — Contracts | ✅ | Pydantic schemas · Postgres tables · `.env` secrets management |
| 1 — MLflow | ✅ | Experiment tracking · `ChurnEnsemble` pyfunc model · quality gate auto-transition |
| 2 — Feature Store | ✅ | Batch pipeline · 48 feature columns · idempotent writes · dry-run: 970,957 valid rows |
| 3 — FastAPI | ✅ | `/predict` · `/predict/batch` (10K max) · `/health` · `/model/info` · async DB logging |
| 4 — Docker | ✅ | Multi-stage build · 5-service Compose · Jenkins with Docker socket mount |
| 5 — Tests | ✅ | 29 tests passing · MockModelManager · async API client · edge case coverage |
| 6 — Jenkins | ✅ | CI (lint + test + build) · CD (deploy + smoke) · monthly retrain · daily monitoring |
| 7 — Monitoring | ✅ | Evidently drift · performance vs ground truth · Grafana dashboard · GitHub Issue alerts |

---

## Key Engineering Decisions

**Temporal split over random split** — Training on Feb, validating on Mar, predicting Apr. A random split gave inflated AUC 0.990 (data leakage via date-correlated features). The honest temporal split gives AUC 0.869.

**Feature Store pattern** — The 28 GB user logs file cannot be queried per-request. Features are pre-computed monthly and stored in Postgres. The API reads from `features_monthly`, not from raw files.

**ModelManager singleton with MLflow → joblib fallback** — The API loads the model once at startup. It tries MLflow Registry first; if the server is unavailable it falls back to local `.joblib` files. This allows development without Docker running.

**Jenkins over GitHub Actions** — Everything is local. Jenkins containers share the host Docker socket and can reach Postgres and MLflow directly, which GitHub Actions runners cannot do without tunneling.

---

## Data Overview

| File | Records | Period | Description |
| --- | --- | --- | --- |
| `train.csv` | 992,931 | Feb 2017 | Training labels |
| `train_v2.csv` | 970,960 | Mar 2017 | Validation labels |
| `sample_submission_v2.csv` | 907,471 | Apr 2017 | Test set (no labels) |
| `transactions.csv` | 21.5M | 2015–2017 | Payment and plan history |
| `user_logs.csv` | 392M rows (~28 GB) | 2015–2017 | Daily listening behavior |

---

---

## Versión en Español

Proyecto end-to-end de predicción de churn sobre el dataset de la competencia [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge). Implementa un pipeline MLOps completo con reentrenamiento automático, monitoreo de drift, CI/CD y una API REST lista para producción local.

### Resultados

Mejor score público: **0.23412 log loss** · Ensemble de LightGBM + XGBoost + CatBoost.

### Stack

```text
Entrenamiento    LightGBM · XGBoost · CatBoost · Optuna
Tracking         MLflow (experimentos + model registry)
API              FastAPI + PostgreSQL
Contenedores     Docker + Docker Compose (5 servicios)
CI/CD            Jenkins — 4 pipelines automatizados
Monitoreo        Evidently AI + Grafana + alertas automáticas
Tests            29 tests (unitarios + integración) — pytest
```

### Levantar el stack

```bash
cp .env.example .env
docker compose up --build -d
# API en http://localhost:8000/docs
```

### Fases completadas

Todas las fases del pipeline están implementadas y funcionales — ver tabla [MLOps Phases](#mlops-phases) arriba.

### Decisiones técnicas clave

- **Split temporal honesto**: train(Feb) → validate(Mar) → predict(Apr). El split random daba AUC 0.990 inflado por data leakage.
- **Feature Store**: los 28 GB de logs no se pueden procesar por request. Features pre-computadas mensualmente en Postgres.
- **Jenkins local**: accede directamente a Docker, Postgres y MLflow del host — GitHub Actions requeriría tunelado.
- **Quality gate automático**: si LogLoss > 0.240 en validación, el modelo no se promueve a Staging.
