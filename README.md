# KKBox Churn Prediction — End-to-End MLOps Pipeline

> End-to-end MLOps pipeline for customer churn prediction on the KKBox music streaming dataset.  
> Demonstrates production-grade ML practices on 1 M users: feature engineering, stacked ensemble,  
> model registry, REST API with interactive SPA, drift detection, and automated CI/CD.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow)
![MLflow](https://img.shields.io/badge/MLflow-3.11.1-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-teal?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Jenkins](https://img.shields.io/badge/CI%2FCD-Jenkins-red?logo=jenkins)
![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)

> **English** (primary) · [Español](#versión-en-español) (secondary)

---

## Table of Contents

- [What This Demonstrates](#what-this-demonstrates)
- [Model Results](#model-results)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Makefile Commands](#makefile-commands)
- [Project Structure](#project-structure)
- [MLOps Phases](#mlops-phases)
- [Key Engineering Decisions](#key-engineering-decisions)
- [MLOps Practices](#mlops-practices)
- [Environment Notes](#environment-notes)
- [Versión en Español](#versión-en-español)

---

## What This Demonstrates

| Skill Area | Implementation |
|---|---|
| **ML Engineering** | LightGBM + XGBoost + CatBoost ensemble · Optuna HPO (60 trials × 3-fold CV) · honest temporal validation split |
| **Feature Engineering** | 48 features across transactions, membership, listening behavior, and membership expiry signals |
| **MLOps** | MLflow experiment tracking + model registry · `ChurnEnsemble` pyfunc · quality gate auto-promotion |
| **Software Engineering** | FastAPI service · Pydantic v2 contracts · async SQLAlchemy · 29-test suite (unit + integration) |
| **Infrastructure** | Docker multi-stage build · Docker Compose (5 services) · PostgreSQL as feature store |
| **CI/CD** | Jenkins 4 pipelines: CI on every commit · CD on merge · monthly retrain · daily monitoring |
| **Monitoring** | Evidently AI drift detection · performance tracking vs ground truth · Grafana dashboards · automated GitHub Issue alerts |
| **Frontend** | Dark-theme SPA (Predict · Batch · Dashboard) · progressive form disclosure · animated gauge · CSV batch upload |

---

## Model Results

Best public score: **0.23412 log loss** (top ~30% of competition leaderboard at close).

### Kaggle Submission History

| Submission | Public Score | Private Score | Description |
|---|---|---|---|
| v1 — Tuned LightGBM | 0.37856 | — | Baseline, 30 features, random split (data leakage) |
| v3 — Temporal fix | 0.30398 | — | Fixed `days_since_last` date-offset bug |
| v4 — Expiry features | 0.23528 | — | +6 membership expiry signals |
| v5 — Best single model | 0.23504 | 0.23494 | 48 features, honest temporal split |
| v8 — CatBoost | 0.23986 | 0.23963 | Single CatBoost, 48 features |
| v9 — LGBM + XGB blend | 0.23975 | 0.23945 | 2-model blend |
| **v10 — 3-model blend** | **0.23436** | **0.23412** | LightGBM + XGBoost + CatBoost |

Key insight: `days_until_expire` is the strongest single predictor — an expired membership with `cancel_at_expire=1` is near-certain churn.

### Optuna Tuning (notebook 07 — 60 trials × 3-fold CV)

| Metric | Baseline | Tuned |
|---|---|---|
| ROC-AUC | 0.9853 | 0.9854 |
| PR-AUC | 0.8549 | 0.8561 |
| F1 (threshold 0.89–0.90) | 0.7599 | 0.7605 |

Best parameters: `n_estimators=705`, `learning_rate=0.019`, `num_leaves=178`, `max_depth=10`. Marginal gain — the baseline was already near-optimal.

### Error Analysis (notebook 08 — 198,587 test users)

| Outcome | Count | Note |
|---|---|---|
| TN — correct renewal | 183,227 | |
| TP — churn correctly detected | 9,412 | |
| FP — false alarm | 2,666 | High cancel signal but renewed |
| FN — missed churner | 3,282 | `days_since_last` avg 5 vs 47 for TP — still active when they churned |

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           Docker Compose                             │
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │
│  │   FastAPI     │  │    MLflow     │  │   Jenkins     │            │
│  │   :8000       │  │   :5001       │  │   :8080       │            │
│  │   + SPA       │  │   + Registry  │  │   4 pipelines │            │
│  └──────┬────────┘  └──────┬────────┘  └──────┬────────┘            │
│         │                  │                  │                     │
│  ┌──────▼──────────────────▼──────────────────▼──────────────────┐  │
│  │                      PostgreSQL :5432                          │  │
│  │  features_monthly · predictions · model_versions ·            │  │
│  │  ground_truth · drift_reports                                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────┐                                                   │
│  │   Grafana     │  ← drift score · churn rate · model version      │
│  │   :3000       │                                                   │
│  └───────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Prediction flow:**

1. Monthly batch pipeline computes 48 features from raw data → stores in `features_monthly`
2. FastAPI `/predict` and `/predict/batch` serve real-time predictions (model loaded once at startup from MLflow Registry)
3. Jenkins retrain pipeline runs on the 1st of each month: compute → train → quality gate → promote to Production
4. Daily monitoring pipeline checks data drift and model performance → opens a GitHub Issue if thresholds exceeded

---

## Tech Stack

### Machine Learning

| Library | Version | Role |
|---|---|---|
| LightGBM | 4.6.0 | Gradient boosting — primary ensemble member |
| XGBoost | 3.2.0 | Gradient boosting — second ensemble member |
| CatBoost | 1.2.10 | Gradient boosting — third ensemble member |
| scikit-learn | 1.8.0 | Preprocessing, cross-validation, metrics |
| Optuna | 4.8.0 | Bayesian hyperparameter optimization (TPE sampler) |
| imbalanced-learn | — | Class imbalance handling |
| SHAP | — | Feature importance and explainability |

### MLOps & Tracking

| Tool | Version | Role |
|---|---|---|
| MLflow | 3.11.1 | Experiment tracking · model registry · artifact store |
| Evidently AI | — | Data drift detection (PSI, DataDriftPreset) |
| Grafana | 10.4 | Monitoring dashboards |

### Serving & API

| Tool | Version | Role |
|---|---|---|
| FastAPI | 0.136.1 | REST API framework |
| Pydantic | 2.13.3 | Request/response validation (v2 contracts) |
| SQLAlchemy | 2.0.49 | Async ORM for prediction logging |
| asyncpg | — | Async PostgreSQL driver |
| aiofiles | — | Static file serving for SPA |

### Infrastructure & CI/CD

| Tool | Role |
|---|---|
| Docker + Docker Compose | Containerisation of all 5 services |
| PostgreSQL 15 | Feature store · prediction log · model version metadata |
| Jenkins | 4 pipelines: CI · CD · monthly retrain · daily monitoring |
| pytest + pytest-asyncio | 29 tests (16 unit + 13 integration) |

---

## Dataset

| File | Records | Period | Description |
|---|---|---|---|
| `train.csv` | 992,931 | Feb 2017 | Training labels |
| `train_v2.csv` | 970,960 | Mar 2017 | Validation labels |
| `sample_submission_v2.csv` | 907,471 | Apr 2017 | Test set (no labels) |
| `transactions.csv` | 21.5 M rows | 2015–2017 | Payment and plan history |
| `user_logs.csv` | 392 M rows (~28 GB) | 2015–2017 | Daily listening behavior |

> Raw data is read-only under `data/raw/`. All transformations produce outputs in `data/processed/`.

**Temporal split** — training on Feb 2017, validation on Mar 2017, test prediction on Apr 2017. A random split produced an inflated AUC of 0.990 due to date-correlated feature leakage. The honest temporal split yields AUC 0.869.

---

## Feature Engineering

48 features engineered from three raw tables, grouped into six signal categories:

| Group | Count | Key signals |
|---|---|---|
| **Transactions** | 15 | `n_transactions`, `n_cancels`, `last_is_cancel`, `price_trend`, `n_payment_methods` |
| **Membership** | 7 | `tenure_days`, `city`, `registered_via`, `age`, `gender_enc` |
| **Listening** | 8 | `avg_daily_secs`, `completion_ratio`, `days_since_last`, `listening_trend` |
| **Expiry & Renewals** | 7 | `days_until_expire`, `is_expired`, `cancel_at_expire`, `auto_renew_at_expire`, `prev_churn` |
| **Recency** | 4 | `days_since_last_tx`, `had_tx_last_7d`, `had_tx_last_30d` |
| **Multi-window Listening** | 6 | `n_days_7d`, `secs_per_day_7d`, `trend_7d_vs_30d` |

The `ChurnEnsemble` pyfunc model wraps all three trained classifiers into a single MLflow artifact. At inference time it averages the three probability outputs before returning a final score.

---

## Quick Start

**Prerequisites:** Docker Desktop, ~6 GB disk, ~4 GB RAM.

```bash
# 1. Clone and configure
git clone https://github.com/josefrodrim/Churn_project.git && cd Churn_project
cp .env.example .env      # defaults work out of the box

# 2. Start all services
docker compose up --build -d

# 3. Verify
curl http://localhost:8000/health
```

| Service | URL | Credentials |
|---|---|---|
| REST API + Swagger UI | http://localhost:8000/docs | `X-API-Key: changeme` |
| SPA Frontend | http://localhost:8000 | `X-API-Key: changeme` |
| MLflow UI | http://localhost:5001 | — |
| Jenkins | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| PostgreSQL | localhost:5432 | see `.env` |

---

## API Reference

OpenAPI docs auto-generated at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Score a single user |
| `POST` | `/predict/batch` | Score up to 10,000 users |
| `GET` | `/predict/{msno}` | Retrieve stored prediction for a user |
| `GET` | `/health` | Liveness check + model status |
| `GET` | `/model/info` | Active model version and metadata |
| `GET` | `/predictions/recent` | Last N predictions (used by SPA dashboard) |

### Single prediction

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{
    "features": {
      "msno": "user_abc",
      "days_until_expire": -30,
      "is_expired": 1,
      "cancel_at_expire": 1,
      "auto_renew_at_expire": 0,
      "last_is_cancel": 1,
      "prev_churn": 1,
      "days_since_last": 45,
      "tenure_days": 120,
      ...
    }
  }'
```

### Response

```json
{
  "msno": "user_abc",
  "churn_prob": 0.989145,
  "churn_label": 1,
  "model_version": "1",
  "predicted_at": "2025-05-01T12:30:00Z"
}
```

---

## Makefile Commands

```bash
make up          # Start full stack (docker compose up --build -d)
make down        # Stop stack
make test        # Run pytest inside the API container
make retrain     # Full retrain cycle: compute features → train → register → promote
make predict     # Monthly batch prediction for current period
make drift       # Generate Evidently drift report
make promote     # Promote Staging model to Production in MLflow
make logs        # Tail API logs
make pipeline    # End-to-end: compute features + retrain + predict
```

---

## Project Structure

```text
src/
  api/            FastAPI service — schemas, config, routes, dependencies, SPA frontend
  models/         Training scripts (numbered 06–17, each a new experiment iteration)
  pipeline/       Batch feature computation + batch prediction (CLI)
  monitoring/     Drift detection, performance tracking, weekly HTML reports
infra/
  docker/         Dockerfiles (API multi-stage, MLflow, Jenkins)
  postgres/       Schema SQL — 5 production tables
  jenkins/        Jenkinsfile.ci · Jenkinsfile.cd · Jenkinsfile.retrain · Jenkinsfile.monitoring
  grafana/        Dashboard JSON + provisioning config
tests/
  unit/           Feature engineering + Pydantic validation (16 tests)
  integration/    API endpoints — auth, predict, batch, health (13 tests)
notebooks/        EDA, feature engineering, Optuna tuning, error analysis
references/       Original competition labelling code (WSDMChurnLabeller.scala)
submissions/      Kaggle CSV history with scores
```

---

## MLOps Phases

| Phase | Status | Description |
|---|---|---|
| 0 — Contracts | ✅ | Pydantic schemas · 5 Postgres tables · `.env` secrets management |
| 1 — MLflow | ✅ | Experiment tracking · `ChurnEnsemble` pyfunc · quality gate auto-promotion |
| 2 — Feature Store | ✅ | Batch pipeline · 48 feature columns · idempotent writes · 970,957 valid rows |
| 3 — FastAPI | ✅ | `/predict` · `/predict/batch` (10K max) · `/health` · `/model/info` · async DB logging |
| 4 — Docker | ✅ | Multi-stage build · 5-service Compose · Jenkins with Docker socket mount |
| 5 — Tests | ✅ | 29 tests passing · MockModelManager · async API client · edge case coverage |
| 6 — Jenkins | ✅ | CI (lint + test + build) · CD (deploy + smoke) · monthly retrain · daily monitoring |
| 7 — Monitoring | ✅ | Evidently drift · performance vs ground truth · Grafana · GitHub Issue alerts |
| 8 — Frontend SPA | ✅ | Predict · Batch CSV · Dashboard · progressive disclosure · animated gauge |

---

## Key Engineering Decisions

**Temporal split over random split** — Training on Feb 2017, validating on Mar 2017. A random split produced an inflated AUC of 0.990 due to data leakage through date-correlated features (`days_since_last`, `days_until_expire`). The honest temporal split yields AUC 0.869 — a realistic estimate of production performance.

**Feature Store pattern** — The 28 GB user logs file cannot be queried per-request. Features are pre-computed monthly and stored in PostgreSQL. The API reads from `features_monthly`, not from raw files. This keeps P99 latency under 50 ms regardless of dataset size.

**ModelManager singleton with MLflow → joblib fallback** — The API loads the model once at startup. It tries MLflow Registry first; if unavailable (local dev without Docker), it falls back to local `.joblib` files. MLflow 3.x DNS rebinding protection requires `--allowed-hosts mlflow,mlflow:5000` when called from inside the Docker network.

**Jenkins over GitHub Actions** — Everything is local. Jenkins containers share the host Docker socket and can reach Postgres and MLflow directly, which GitHub Actions hosted runners cannot do without tunneling or mock infrastructure.

**Class imbalance at 6.4% churn rate** — Models trained with `scale_pos_weight` (LightGBM/XGBoost). The F1-optimal threshold (≈0.89) is calibrated post-training rather than fixed at 0.5.

**3-model blend over single model** — Moving from the best single model (log loss 0.23504) to the 3-model blend (0.23412) gives a consistent −0.04% gain from diversity in base learners.

---

## MLOps Practices

| Practice | Implementation |
|---|---|
| **Reproducibility** | Numbered training scripts (06–17) track every experiment iteration with full params logged to MLflow |
| **Temporal validation** | Time-based split prevents leakage from date-correlated features |
| **Experiment tracking** | MLflow logs every run: params, metrics, artifacts, model signature |
| **Model registry** | MLflow stages: None → Staging → Production with full version lineage |
| **Auto-promotion** | Quality gate (LogLoss ≤ 0.240 · ROC-AUC ≥ 0.85) triggers auto-transition to Production |
| **Serialized ensemble** | `ChurnEnsemble` pyfunc wraps 3 models as one registry artifact — no training/serving skew |
| **Feature store** | Pre-computed monthly features in PostgreSQL — decouples dataset scale from API latency |
| **REST API** | FastAPI with Pydantic v2 validation, batch endpoint (10K max), async prediction logging |
| **Interactive frontend** | Dark-theme SPA with progressive form disclosure, preset scenarios, and animated conic gauge |
| **Drift detection** | Evidently AI DataDriftPreset: baseline Feb vs current period, HTML + JSON sidecar reports |
| **Containerisation** | All 5 services in Docker Compose; API uses multi-stage build (builder + slim runtime) |
| **CI/CD** | Jenkins 4-pipeline setup: lint → test → build → deploy → smoke test → drift report |
| **Test coverage** | 29 tests: Pydantic validation, MockModelManager, async HTTP client, auth, edge cases |
| **Automated alerts** | Drift > 0.15 or AUC < 0.85 → GitHub Issue opened automatically |

---

## Environment Notes

- MLflow runs on port **5001** (not 5000) — macOS AirPlay/ControlCenter occupies 5000 on the host. Inside the Docker network the service still listens on 5000.
- MLflow 3.x DNS rebinding protection requires `--allowed-hosts mlflow,mlflow:5000,localhost,localhost:5001` in the server command. Without this, inter-container requests are rejected with 403.
- The API container mounts the `mlflow_artifacts` volume so model registration can write artifacts without permission errors.
- After editing source files, rebuild with `docker compose up --build -d` or use `docker cp` for hot-patching without a full rebuild.

---

---

## Versión en Español

> [English](#kkbox-churn-prediction--end-to-end-mlops-pipeline) (primario) · **Español** (secundario)

Pipeline MLOps completo para predicción de churn de clientes sobre el dataset de KKBox (streaming de música).  
Demuestra prácticas de ML en producción sobre 1 M de usuarios: feature engineering, ensemble apilado,  
model registry, API REST con SPA interactiva, detección de drift y CI/CD automatizado.

---

## Tabla de Contenidos

- [Qué demuestra](#qué-demuestra)
- [Resultados del modelo](#resultados-del-modelo)
- [Arquitectura](#arquitectura)
- [Stack tecnológico](#stack-tecnológico)
- [Dataset](#dataset-1)
- [Feature Engineering](#feature-engineering-1)
- [Inicio rápido](#inicio-rápido)
- [Referencia de la API](#referencia-de-la-api)
- [Comandos del Makefile](#comandos-del-makefile)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Fases MLOps](#fases-mlops)
- [Decisiones técnicas clave](#decisiones-técnicas-clave)
- [Prácticas MLOps](#prácticas-mlops)
- [Notas de entorno](#notas-de-entorno)

---

## Qué demuestra

| Área | Implementación |
|---|---|
| **ML Engineering** | Ensemble LightGBM + XGBoost + CatBoost · Optuna HPO (60 trials × 3-fold CV) · split temporal honesto |
| **Feature Engineering** | 48 features: transacciones, membresía, comportamiento de escucha y señales de vencimiento |
| **MLOps** | MLflow tracking + model registry · `ChurnEnsemble` pyfunc · quality gate con auto-promoción |
| **Ingeniería de software** | Servicio FastAPI · contratos Pydantic v2 · SQLAlchemy async · suite de 29 tests |
| **Infraestructura** | Docker multi-stage · Docker Compose (5 servicios) · PostgreSQL como feature store |
| **CI/CD** | Jenkins 4 pipelines: CI en cada commit · CD en merge · retrain mensual · monitoreo diario |
| **Monitoreo** | Drift con Evidently AI · tracking de performance vs ground truth · dashboards Grafana · alertas automáticas |
| **Frontend** | SPA tema oscuro (Predict · Batch · Dashboard) · divulgación progresiva del formulario · gauge animado · batch CSV |

---

## Resultados del modelo

Mejor score público: **0.23412 log loss** (top ~30% del leaderboard al cierre de la competencia).

### Historial de submissions en Kaggle

| Submission | Score público | Score privado | Descripción |
|---|---|---|---|
| v1 — LightGBM tuneado | 0.37856 | — | Baseline, 30 features, split random (data leakage) |
| v3 — Fix temporal | 0.30398 | — | Corregido bug de offset en `days_since_last` |
| v4 — Features de vencimiento | 0.23528 | — | +6 señales de membresía al vencimiento |
| v5 — Mejor modelo individual | 0.23504 | 0.23494 | 48 features, split temporal honesto |
| v8 — CatBoost | 0.23986 | 0.23963 | CatBoost solo, 48 features |
| v9 — Blend LGBM + XGB | 0.23975 | 0.23945 | Blend de 2 modelos |
| **v10 — Blend 3 modelos** | **0.23436** | **0.23412** | LightGBM + XGBoost + CatBoost |

Insight clave: `days_until_expire` es el predictor más fuerte — una membresía vencida con `cancel_at_expire=1` es churn casi seguro.

### Tuning con Optuna (notebook 07 — 60 trials × 3-fold CV)

| Métrica | Baseline | Tuneado |
|---|---|---|
| ROC-AUC | 0.9853 | 0.9854 |
| PR-AUC | 0.8549 | 0.8561 |
| F1 (umbral 0.89–0.90) | 0.7599 | 0.7605 |

Mejores parámetros: `n_estimators=705`, `learning_rate=0.019`, `num_leaves=178`, `max_depth=10`. La mejora es marginal — el modelo base ya estaba cerca del óptimo.

### Análisis de errores (notebook 08 — 198,587 usuarios de test)

| Resultado | Cantidad | Nota |
|---|---|---|
| TN — renewal correcto | 183,227 | |
| TP — churn detectado | 9,412 | |
| FP — falsa alarma | 2,666 | Señal de cancelación alta pero renovó |
| FN — churner no detectado | 3,282 | `days_since_last` promedio 5 vs 47 en TP — seguían activos cuando churnaron |

---

## Arquitectura

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           Docker Compose                             │
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │
│  │   FastAPI     │  │    MLflow     │  │   Jenkins     │            │
│  │   :8000       │  │   :5001       │  │   :8080       │            │
│  │   + SPA       │  │   + Registry  │  │   4 pipelines │            │
│  └──────┬────────┘  └──────┬────────┘  └──────┬────────┘            │
│         │                  │                  │                     │
│  ┌──────▼──────────────────▼──────────────────▼──────────────────┐  │
│  │                      PostgreSQL :5432                          │  │
│  │  features_monthly · predictions · model_versions ·            │  │
│  │  ground_truth · drift_reports                                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────┐                                                   │
│  │   Grafana     │  ← drift score · churn rate · model version      │
│  │   :3000       │                                                   │
│  └───────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Flujo de predicción:**

1. El pipeline batch mensual computa 48 features desde los datos crudos → los almacena en `features_monthly`
2. FastAPI `/predict` y `/predict/batch` sirven predicciones en tiempo real (modelo cargado una vez al arranque desde MLflow Registry)
3. El pipeline de reentrenamiento Jenkins corre el día 1 de cada mes: compute → train → quality gate → promote a Production
4. El pipeline de monitoreo diario verifica drift y performance → abre un Issue en GitHub si se superan los umbrales

---

## Stack tecnológico

### Machine Learning

| Librería | Versión | Rol |
|---|---|---|
| LightGBM | 4.6.0 | Gradient boosting — miembro principal del ensemble |
| XGBoost | 3.2.0 | Gradient boosting — segundo miembro del ensemble |
| CatBoost | 1.2.10 | Gradient boosting — tercer miembro del ensemble |
| scikit-learn | 1.8.0 | Preprocesamiento, validación cruzada, métricas |
| Optuna | 4.8.0 | Optimización bayesiana de hiperparámetros (TPE sampler) |
| imbalanced-learn | — | Manejo del desbalance de clases |
| SHAP | — | Importancia de features y explicabilidad |

### MLOps y Tracking

| Herramienta | Versión | Rol |
|---|---|---|
| MLflow | 3.11.1 | Tracking de experimentos · model registry · artifact store |
| Evidently AI | — | Detección de drift (PSI, DataDriftPreset) |
| Grafana | 10.4 | Dashboards de monitoreo |

### Serving y API

| Herramienta | Versión | Rol |
|---|---|---|
| FastAPI | 0.136.1 | Framework REST API |
| Pydantic | 2.13.3 | Validación de requests/responses (contratos v2) |
| SQLAlchemy | 2.0.49 | ORM async para logging de predicciones |
| asyncpg | — | Driver PostgreSQL asíncrono |
| aiofiles | — | Servicio de archivos estáticos para SPA |

### Infraestructura y CI/CD

| Herramienta | Rol |
|---|---|
| Docker + Docker Compose | Contenedorización de los 5 servicios |
| PostgreSQL 15 | Feature store · log de predicciones · metadatos de versiones |
| Jenkins | 4 pipelines: CI · CD · retrain mensual · monitoreo diario |
| pytest + pytest-asyncio | 29 tests (16 unitarios + 13 integración) |

---

## Dataset

| Archivo | Registros | Período | Descripción |
|---|---|---|---|
| `train.csv` | 992,931 | Feb 2017 | Etiquetas de entrenamiento |
| `train_v2.csv` | 970,960 | Mar 2017 | Etiquetas de validación |
| `sample_submission_v2.csv` | 907,471 | Abr 2017 | Set de test (sin etiquetas) |
| `transactions.csv` | 21.5 M filas | 2015–2017 | Historial de pagos y planes |
| `user_logs.csv` | 392 M filas (~28 GB) | 2015–2017 | Comportamiento de escucha diario |

> Los datos crudos son de solo lectura en `data/raw/`. Todas las transformaciones producen outputs en `data/processed/`.

**Split temporal** — entrenamiento en Feb 2017, validación en Mar 2017, predicción en Abr 2017. Un split random producía AUC inflado de 0.990 por data leakage en features correlacionadas con la fecha. El split temporal honesto da AUC 0.869.

---

## Feature Engineering

48 features construidas a partir de tres tablas crudas, agrupadas en seis categorías:

| Grupo | Cantidad | Señales clave |
|---|---|---|
| **Transacciones** | 15 | `n_transactions`, `n_cancels`, `last_is_cancel`, `price_trend`, `n_payment_methods` |
| **Membresía** | 7 | `tenure_days`, `city`, `registered_via`, `age`, `gender_enc` |
| **Escucha** | 8 | `avg_daily_secs`, `completion_ratio`, `days_since_last`, `listening_trend` |
| **Vencimiento y renovaciones** | 7 | `days_until_expire`, `is_expired`, `cancel_at_expire`, `auto_renew_at_expire`, `prev_churn` |
| **Recencia** | 4 | `days_since_last_tx`, `had_tx_last_7d`, `had_tx_last_30d` |
| **Escucha multi-ventana** | 6 | `n_days_7d`, `secs_per_day_7d`, `trend_7d_vs_30d` |

El modelo pyfunc `ChurnEnsemble` encapsula los tres clasificadores entrenados como un único artefacto de MLflow. En inferencia promedia las tres salidas de probabilidad antes de devolver el score final.

---

## Inicio rápido

**Requisitos previos:** Docker Desktop, ~6 GB de disco, ~4 GB de RAM.

```bash
# 1. Clonar y configurar
git clone https://github.com/josefrodrim/Churn_project.git && cd Churn_project
cp .env.example .env      # los valores por defecto funcionan sin modificar

# 2. Levantar todos los servicios
docker compose up --build -d

# 3. Verificar
curl http://localhost:8000/health
```

| Servicio | URL | Credenciales |
|---|---|---|
| REST API + Swagger UI | http://localhost:8000/docs | `X-API-Key: changeme` |
| SPA Frontend | http://localhost:8000 | `X-API-Key: changeme` |
| MLflow UI | http://localhost:5001 | — |
| Jenkins | http://localhost:8080 | admin / admin |
| Grafana | http://localhost:3000 | admin / admin |
| PostgreSQL | localhost:5432 | ver `.env` |

---

## Referencia de la API

Documentación OpenAPI autogenerada en `http://localhost:8000/docs`.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/predict` | Predice para un usuario |
| `POST` | `/predict/batch` | Predice para hasta 10,000 usuarios |
| `GET` | `/predict/{msno}` | Recupera predicción almacenada de un usuario |
| `GET` | `/health` | Liveness check + estado del modelo |
| `GET` | `/model/info` | Versión activa del modelo y metadatos |
| `GET` | `/predictions/recent` | Últimas N predicciones (usadas por el dashboard SPA) |

### Predicción individual

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: changeme" \
  -d '{
    "features": {
      "msno": "user_abc",
      "days_until_expire": -30,
      "is_expired": 1,
      "cancel_at_expire": 1,
      "auto_renew_at_expire": 0,
      "last_is_cancel": 1,
      "prev_churn": 1,
      "days_since_last": 45,
      "tenure_days": 120,
      ...
    }
  }'
```

### Respuesta

```json
{
  "msno": "user_abc",
  "churn_prob": 0.989145,
  "churn_label": 1,
  "model_version": "1",
  "predicted_at": "2025-05-01T12:30:00Z"
}
```

---

## Comandos del Makefile

```bash
make up          # Levantar el stack completo (docker compose up --build -d)
make down        # Detener el stack
make test        # Ejecutar pytest dentro del contenedor de la API
make retrain     # Ciclo completo: compute features → train → registrar → promover
make predict     # Predicción batch mensual para el período actual
make drift       # Generar reporte de drift con Evidently
make promote     # Promover modelo de Staging a Production en MLflow
make logs        # Ver logs de la API en tiempo real
make pipeline    # End-to-end: compute features + retrain + predict
```

---

## Estructura del proyecto

```text
src/
  api/            Servicio FastAPI — schemas, config, rutas, dependencias, SPA frontend
  models/         Scripts de entrenamiento (numerados 06–17, cada uno un experimento)
  pipeline/       Cómputo de features batch + predicción batch (CLI)
  monitoring/     Detección de drift, tracking de performance, reportes HTML semanales
infra/
  docker/         Dockerfiles (API multi-stage, MLflow, Jenkins)
  postgres/       SQL del schema — 5 tablas de producción
  jenkins/        Jenkinsfile.ci · Jenkinsfile.cd · Jenkinsfile.retrain · Jenkinsfile.monitoring
  grafana/        JSON del dashboard + configuración de provisioning
tests/
  unit/           Feature engineering + validación Pydantic (16 tests)
  integration/    Endpoints de la API — auth, predict, batch, health (13 tests)
notebooks/        EDA, feature engineering, tuning con Optuna, análisis de errores
references/       Código original de labelling de la competencia (WSDMChurnLabeller.scala)
submissions/      Historial de CSVs de Kaggle con scores
```

---

## Fases MLOps

| Fase | Estado | Descripción |
|---|---|---|
| 0 — Contratos | ✅ | Schemas Pydantic · 5 tablas Postgres · gestión de secrets con `.env` |
| 1 — MLflow | ✅ | Tracking de experimentos · `ChurnEnsemble` pyfunc · quality gate con auto-promoción |
| 2 — Feature Store | ✅ | Pipeline batch · 48 columnas de features · escrituras idempotentes · 970,957 filas válidas |
| 3 — FastAPI | ✅ | `/predict` · `/predict/batch` (máx 10K) · `/health` · `/model/info` · logging async en DB |
| 4 — Docker | ✅ | Build multi-stage · Compose con 5 servicios · Jenkins con socket Docker montado |
| 5 — Tests | ✅ | 29 tests pasando · MockModelManager · cliente API async · cobertura de edge cases |
| 6 — Jenkins | ✅ | CI (lint + test + build) · CD (deploy + smoke) · retrain mensual · monitoreo diario |
| 7 — Monitoreo | ✅ | Drift Evidently · performance vs ground truth · Grafana · alertas por GitHub Issues |
| 8 — Frontend SPA | ✅ | Predict · Batch CSV · Dashboard · divulgación progresiva · gauge animado |

---

## Decisiones técnicas clave

**Split temporal en lugar de split random** — Entrenamiento en Feb 2017, validación en Mar 2017. Un split random producía AUC inflado de 0.990 por data leakage en features correlacionadas con la fecha (`days_since_last`, `days_until_expire`). El split temporal honesto da AUC 0.869 — estimación realista del rendimiento en producción.

**Patrón Feature Store** — El archivo de logs tiene 28 GB y no puede consultarse por request. Las features se pre-computan mensualmente y se almacenan en PostgreSQL. La API lee desde `features_monthly`, no desde los archivos crudos. Esto mantiene la latencia P99 bajo 50 ms independientemente del tamaño del dataset.

**Singleton ModelManager con fallback MLflow → joblib** — La API carga el modelo una sola vez al arranque. Intenta primero el MLflow Registry; si no está disponible (desarrollo local sin Docker), cae a archivos `.joblib` locales. La protección anti-DNS rebinding de MLflow 3.x requiere `--allowed-hosts mlflow,mlflow:5000` cuando se llama desde dentro de la red Docker.

**Jenkins en lugar de GitHub Actions** — Todo es local. Los contenedores Jenkins comparten el socket Docker del host y pueden conectar directamente a Postgres y MLflow, algo que los runners hosteados de GitHub Actions no pueden hacer sin tunelado o infraestructura mock.

**Manejo del desbalance de clases al 6.4%** — Modelos entrenados con `scale_pos_weight` (LightGBM/XGBoost). El umbral óptimo para F1 (≈0.89) se calibra post-entrenamiento en lugar de fijarse en 0.5.

**Blend de 3 modelos en lugar de modelo individual** — Pasar del mejor modelo individual (log loss 0.23504) al blend de 3 (0.23412) aporta una ganancia consistente de −0.04% gracias a la diversidad entre los base learners.

---

## Prácticas MLOps

| Práctica | Implementación |
|---|---|
| **Reproducibilidad** | Scripts de entrenamiento numerados (06–17) con todos los params loggeados en MLflow |
| **Validación temporal** | Split por tiempo previene leakage de features correlacionadas con la fecha |
| **Tracking de experimentos** | MLflow registra cada run: params, métricas, artefactos, firma del modelo |
| **Model registry** | MLflow stages: None → Staging → Production con linaje completo de versiones |
| **Auto-promoción** | Quality gate (LogLoss ≤ 0.240 · ROC-AUC ≥ 0.85) dispara la transición automática a Production |
| **Ensemble serializado** | `ChurnEnsemble` pyfunc encapsula 3 modelos como un artefacto — sin training/serving skew |
| **Feature store** | Features mensuales pre-computadas en PostgreSQL — desacopla escala del dataset de la latencia de la API |
| **REST API** | FastAPI con validación Pydantic v2, endpoint batch (máx 10K), logging async de predicciones |
| **Frontend interactivo** | SPA tema oscuro con divulgación progresiva del formulario, presets y gauge cónico animado |
| **Detección de drift** | Evidently AI DataDriftPreset: baseline Feb vs período actual, reportes HTML + JSON |
| **Contenedorización** | 5 servicios en Docker Compose; API con build multi-stage (builder + slim runtime) |
| **CI/CD** | Jenkins 4 pipelines: lint → test → build → deploy → smoke test → drift report |
| **Cobertura de tests** | 29 tests: validación Pydantic, MockModelManager, cliente HTTP async, auth, edge cases |
| **Alertas automáticas** | Drift > 0.15 o AUC < 0.85 → Issue en GitHub abierto automáticamente |

---

## Notas de entorno

- MLflow corre en el puerto **5001** (no 5000) — macOS AirPlay/ControlCenter ocupa el 5000 en el host. Dentro de la red Docker el servicio escucha en 5000.
- La protección anti-DNS rebinding de MLflow 3.x requiere `--allowed-hosts mlflow,mlflow:5000,localhost,localhost:5001` en el comando del servidor. Sin esto, las peticiones inter-contenedor son rechazadas con 403.
- El contenedor de la API monta el volumen `mlflow_artifacts` para que el registro de modelos pueda escribir artefactos sin errores de permisos.
- Al modificar archivos fuente, reconstruir con `docker compose up --build -d` o usar `docker cp` para hot-patching sin rebuild completo.
