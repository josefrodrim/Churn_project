# KKBox Churn Prediction — End-to-End MLOps Pipeline

> **English** (primary) · [Español](#versión-en-español) (secondary)

A production-grade machine learning system built on the [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) dataset. The project goes beyond model training — it implements a complete local MLOps stack with automated retraining, drift monitoring, CI/CD pipelines, and a REST API ready for deployment.

---

## What This Demonstrates

| Skill Area | Implementation |
| --- | --- |
| **ML Engineering** | LightGBM + XGBoost + CatBoost ensemble · Optuna hyperparameter tuning · honest temporal validation split |
| **Feature Engineering** | 48 features across transactions, membership, listening behavior, and expiry signals |
| **MLOps** | MLflow experiment tracking + model registry · automated quality gates · model version promotion |
| **Software Engineering** | FastAPI service · Pydantic v2 contracts · async SQLAlchemy · 29-test suite (unit + integration) |
| **Infrastructure** | Docker multi-stage build · Docker Compose (5 services) · PostgreSQL as feature store |
| **CI/CD** | Jenkins pipelines: CI on every commit, CD on merge to main, monthly retrain, daily monitoring |
| **Monitoring** | Evidently AI data drift · performance tracking against ground truth · Grafana dashboards · automated alerts |

---

## Model Results

Best public score: **0.23412 log loss** (top ~30% of competition leaderboard at close).

| Submission | Public Score | Private Score | Description |
| --- | --- | --- | --- |
| v1 — Tuned LightGBM | 0.37856 | — | Baseline, 30 features, random split (overfit due to leakage) |
| v3 — Temporal fix | 0.30398 | — | Fixed `days_since_last` date-offset bug |
| v4 — Expiry features | 0.23528 | — | +6 membership expiry signals |
| v5 — Best single model | 0.23504 | 0.23494 | 48 features, honest temporal split |
| v8 — CatBoost | 0.23986 | 0.23963 | Single CatBoost, 48 features |
| v9 — LGBM + XGB blend | 0.23975 | 0.23945 | 2-model blend |
| **v10 — 3-model blend** | **0.23436** | **0.23412** | LightGBM + XGBoost + CatBoost |

Key insight: `days_until_expire` is the strongest single predictor — an expired membership with `cancel_at_expire=1` is almost certain churn.

**Optuna tuning results** (notebook 07, 60 trials × 3-fold CV):

| Metric | Baseline | Tuned |
| --- | --- | --- |
| ROC-AUC | 0.9853 | 0.9854 |
| PR-AUC | 0.8549 | 0.8561 |
| F1 (threshold 0.89–0.90) | 0.7599 | 0.7605 |

Best parameters: `n_estimators=705`, `learning_rate=0.019`, `num_leaves=178`, `max_depth=10`. Marginal gain — the baseline was already near-optimal.

**Error analysis** (notebook 08, test set of 198,587 users):

| Result | Count | Note |
| --- | --- | --- |
| TN — correct renewal | 183,227 | |
| TP — churn correctly detected | 9,412 | |
| FP — false alarm | 2,666 | High cancel signal but renewed |
| FN — missed churner | 3,282 | `days_since_last` avg 5 vs 47 for TP — still active when they churned |

---

## Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                      Docker Compose                         │
│                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ FastAPI  │   │  MLflow  │   │ Jenkins  │               │
│  │  :8000   │   │  :5001   │   │  :8080   │               │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘               │
│       │              │              │                      │
│  ┌────▼──────────────▼──────────────▼──────┐               │
│  │              PostgreSQL :5432            │               │
│  │  features_monthly · predictions ·       │               │
│  │  model_versions · ground_truth ·        │               │
│  │  drift_reports                          │               │
│  └──────────────────────────────────────────┘               │
│                                                            │
│  ┌──────────┐                                              │
│  │ Grafana  │  ← drift score · churn rate · latency       │
│  │  :3000   │                                              │
│  └──────────┘                                              │
└────────────────────────────────────────────────────────────┘
```

**Prediction flow:**

1. Monthly batch pipeline computes 48 features from raw data → stores in `features_monthly`
2. FastAPI `/predict` and `/predict/batch` serve real-time predictions (model loaded once at startup)
3. Jenkins retrain pipeline runs on the 1st of each month: compute → train → quality gate → promote
4. Daily monitoring pipeline checks data drift and model performance → alerts via GitHub Issues if thresholds exceeded

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
| REST API + Swagger UI | <http://localhost:8000/docs> | `X-API-Key: changeme` |
| MLflow UI | <http://localhost:5001> | — |
| Jenkins | <http://localhost:8080> | admin / admin |
| Grafana | <http://localhost:3000> | admin / admin |
| PostgreSQL | localhost:5432 | see `.env` |

> **Note:** MLflow runs on port 5001 (not 5000) because macOS AirPlay occupies 5000 on the host. Inside the Docker network the service still listens on 5000.

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
notebooks/      EDA, feature engineering experiments, model tuning, error analysis
references/     Original competition labelling code (WSDMChurnLabeller.scala)
submissions/    Kaggle CSV history with scores
reports/
  monitoring/   Evidently HTML drift reports (generated at runtime)
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

**Temporal split over random split** — Training on Feb 2017, validating on Mar, predicting Apr. A random split produced inflated AUC 0.990 due to data leakage via date-correlated features. The honest temporal split yields AUC 0.869 — a more realistic estimate of production performance.

**Feature Store pattern** — The 28 GB user logs file cannot be queried per-request. Features are pre-computed monthly and stored in Postgres. The API reads from `features_monthly`, not from raw files. This keeps P99 latency under 50ms regardless of dataset size.

**ModelManager singleton with MLflow → joblib fallback** — The API loads the model once at startup. It tries MLflow Registry first; if the server is unavailable (e.g. local development without Docker), it falls back to local `.joblib` files. DNS rebinding protection in MLflow 2.x requires `--allowed-hosts mlflow` when called from inside the Docker network.

**Jenkins over GitHub Actions** — Everything is local. Jenkins containers share the host Docker socket and can reach Postgres and MLflow directly, which GitHub Actions hosted runners cannot do without tunneling or mock infrastructure.

**Class imbalance handling** — The dataset has a 6.4% churn rate. Models were trained with `scale_pos_weight` (XGBoost/LightGBM) and `class_weight='balanced'` alternatives. The F1-optimal threshold (≈0.89) is calibrated post-training rather than fixed at 0.5.

---

## Data Overview

| File | Records | Period | Description |
| --- | --- | --- | --- |
| `train.csv` | 992,931 | Feb 2017 | Training labels |
| `train_v2.csv` | 970,960 | Mar 2017 | Validation labels |
| `sample_submission_v2.csv` | 907,471 | Apr 2017 | Test set (no labels) |
| `transactions.csv` | 21.5M rows | 2015–2017 | Payment and plan history |
| `user_logs.csv` | 392M rows (~28 GB) | 2015–2017 | Daily listening behavior |

Raw data is read-only under `data/raw/`. All transformations produce outputs in `data/processed/`.

---

---

## Versión en Español

> [English](#kkbox-churn-prediction--end-to-end-mlops-pipeline) (primario) · **Español** (secundario)

Sistema de machine learning de nivel producción construido sobre el dataset del [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge). El proyecto va más allá del entrenamiento de modelos: implementa un stack MLOps completo con reentrenamiento automático, monitoreo de drift, pipelines CI/CD y una API REST lista para despliegue local.

---

## Qué demuestra este proyecto

| Área | Implementación |
| --- | --- |
| **ML Engineering** | Ensemble LightGBM + XGBoost + CatBoost · tuning con Optuna · split temporal honesto |
| **Feature Engineering** | 48 features: transacciones, membresía, comportamiento de escucha y señales de vencimiento |
| **MLOps** | MLflow tracking + model registry · quality gates automáticos · promoción de versiones |
| **Ingeniería de software** | Servicio FastAPI · contratos Pydantic v2 · SQLAlchemy async · suite de 29 tests |
| **Infraestructura** | Docker multi-stage · Docker Compose (5 servicios) · PostgreSQL como feature store |
| **CI/CD** | Jenkins: CI en cada commit, CD en merge a main, reentrenamiento mensual, monitoreo diario |
| **Monitoreo** | Drift de datos con Evidently AI · tracking de performance vs ground truth · dashboards Grafana · alertas automáticas |

---

## Resultados del modelo

Mejor score público: **0.23412 log loss** (top ~30% del leaderboard al cierre de la competencia).

| Submission | Score público | Score privado | Descripción |
| --- | --- | --- | --- |
| v1 — LightGBM tuneado | 0.37856 | — | Baseline, 30 features, split random (sobreajuste por data leakage) |
| v3 — Fix temporal | 0.30398 | — | Corregido bug de offset en `days_since_last` |
| v4 — Features de vencimiento | 0.23528 | — | +6 señales de membresía al vencimiento |
| v5 — Mejor modelo individual | 0.23504 | 0.23494 | 48 features, split temporal honesto |
| v8 — CatBoost | 0.23986 | 0.23963 | CatBoost solo, 48 features |
| v9 — Blend LGBM + XGB | 0.23975 | 0.23945 | Blend de 2 modelos |
| **v10 — Blend 3 modelos** | **0.23436** | **0.23412** | LightGBM + XGBoost + CatBoost |

Insight clave: `days_until_expire` es el predictor más fuerte — una membresía vencida con `cancel_at_expire=1` es churn casi seguro.

**Resultados del tuning con Optuna** (notebook 07, 60 trials × 3-fold CV):

| Métrica | Baseline | Tuneado |
| --- | --- | --- |
| ROC-AUC | 0.9853 | 0.9854 |
| PR-AUC | 0.8549 | 0.8561 |
| F1 (umbral 0.89–0.90) | 0.7599 | 0.7605 |

Mejores parámetros: `n_estimators=705`, `learning_rate=0.019`, `num_leaves=178`, `max_depth=10`. La mejora es marginal — el modelo base ya estaba cerca del óptimo.

**Análisis de errores** (notebook 08, set de test con 198,587 usuarios):

| Resultado | Cantidad | Nota |
| --- | --- | --- |
| TN — renewal correcto | 183,227 | |
| TP — churn detectado | 9,412 | |
| FP — falsa alarma | 2,666 | Señal de cancelación alta pero renovó |
| FN — churner no detectado | 3,282 | `days_since_last` promedio 5 vs 47 en TP — seguían activos cuando churnaron |

---

## Arquitectura

```text
┌────────────────────────────────────────────────────────────┐
│                      Docker Compose                         │
│                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ FastAPI  │   │  MLflow  │   │ Jenkins  │               │
│  │  :8000   │   │  :5001   │   │  :8080   │               │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘               │
│       │              │              │                      │
│  ┌────▼──────────────▼──────────────▼──────┐               │
│  │              PostgreSQL :5432            │               │
│  │  features_monthly · predictions ·       │               │
│  │  model_versions · ground_truth ·        │               │
│  │  drift_reports                          │               │
│  └──────────────────────────────────────────┘               │
│                                                            │
│  ┌──────────┐                                              │
│  │ Grafana  │  ← drift score · churn rate · latencia      │
│  │  :3000   │                                              │
│  └──────────┘                                              │
└────────────────────────────────────────────────────────────┘
```

**Flujo de predicción:**

1. El pipeline batch mensual computa 48 features desde los datos crudos → los almacena en `features_monthly`
2. FastAPI `/predict` y `/predict/batch` sirven predicciones en tiempo real (modelo cargado una vez al arranque)
3. El pipeline de reentrenamiento de Jenkins corre el día 1 de cada mes: compute → train → quality gate → promote
4. El pipeline de monitoreo diario verifica drift de datos y performance del modelo → abre un Issue en GitHub si se superan los umbrales

---

## Stack tecnológico

```text
Entrenamiento    LightGBM · XGBoost · CatBoost · scikit-learn · Optuna
Tracking         MLflow (servidor + model registry + artifact store)
API              FastAPI · Pydantic v2 · SQLAlchemy async · httpx
Base de datos    PostgreSQL 15
Contenedores     Docker (multi-stage) · Docker Compose
CI/CD            Jenkins (4 pipelines: CI, CD, retrain, monitoreo)
Monitoreo        Evidently AI · Grafana 10.4 · psycopg2
Tests            pytest · pytest-asyncio · 29 tests (unitarios + integración)
```

---

## Inicio rápido

```bash
# 1. Clonar y configurar
git clone <repo-url> && cd Churn_project
cp .env.example .env          # editar secrets si es necesario

# 2. Levantar todos los servicios
docker compose up --build -d

# 3. Verificar
curl http://localhost:8000/health
```

| Servicio | URL | Credenciales |
| --- | --- | --- |
| REST API + Swagger UI | <http://localhost:8000/docs> | `X-API-Key: changeme` |
| MLflow UI | <http://localhost:5001> | — |
| Jenkins | <http://localhost:8080> | admin / admin |
| Grafana | <http://localhost:3000> | admin / admin |
| PostgreSQL | localhost:5432 | ver `.env` |

> **Nota:** MLflow corre en el puerto 5001 (no 5000) porque macOS AirPlay ocupa el 5000 en el host. Dentro de la red Docker el servicio escucha en 5000.

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
  api/          Servicio FastAPI — schemas, config, rutas, dependencias
  models/       Scripts de entrenamiento (numerados 06–17, cada uno un experimento)
  pipeline/     Cómputo de features batch + predicción batch (CLI)
  monitoring/   Detección de drift, tracking de performance, reportes HTML semanales
infra/
  docker/       Dockerfiles (API multi-stage, MLflow, Jenkins)
  postgres/     SQL del schema — 5 tablas de producción
  jenkins/      Jenkinsfile.ci · Jenkinsfile.cd · Jenkinsfile.retrain · Jenkinsfile.monitoring
  grafana/      JSON del dashboard + configuración de provisioning
tests/
  unit/         Feature engineering + validación Pydantic (16 tests)
  integration/  Endpoints de la API — auth, predict, batch, health (13 tests)
notebooks/      EDA, experimentos de features, tuning de modelos, análisis de errores
references/     Código original de labelling de la competencia (WSDMChurnLabeller.scala)
submissions/    Historial de CSVs de Kaggle con scores
reports/
  monitoring/   Reportes HTML de drift generados por Evidently (generados en runtime)
```

---

## Fases MLOps

| Fase | Estado | Descripción |
| --- | --- | --- |
| 0 — Contratos | ✅ | Schemas Pydantic · tablas Postgres · gestión de secrets con `.env` |
| 1 — MLflow | ✅ | Tracking de experimentos · modelo `ChurnEnsemble` pyfunc · quality gate con auto-transición |
| 2 — Feature Store | ✅ | Pipeline batch · 48 columnas de features · escrituras idempotentes · dry-run: 970,957 filas válidas |
| 3 — FastAPI | ✅ | `/predict` · `/predict/batch` (máx 10K) · `/health` · `/model/info` · logging async en DB |
| 4 — Docker | ✅ | Build multi-stage · Compose con 5 servicios · Jenkins con socket Docker montado |
| 5 — Tests | ✅ | 29 tests pasando · MockModelManager · cliente API async · cobertura de edge cases |
| 6 — Jenkins | ✅ | CI (lint + test + build) · CD (deploy + smoke test) · retrain mensual · monitoreo diario |
| 7 — Monitoreo | ✅ | Drift Evidently · performance vs ground truth · dashboard Grafana · alertas por GitHub Issues |

---

## Decisiones técnicas clave

**Split temporal en lugar de split random** — Entrenamiento en Feb 2017, validación en Mar, predicción en Abr. Un split random producía un AUC inflado de 0.990 por data leakage en features correlacionadas con la fecha. El split temporal honesto da AUC 0.869 — una estimación más realista del rendimiento en producción.

**Patrón Feature Store** — El archivo de logs de usuarios tiene 28 GB y no puede consultarse por request. Las features se pre-computan mensualmente y se almacenan en Postgres. La API lee desde `features_monthly`, no desde los archivos crudos. Esto mantiene la latencia P99 por debajo de 50ms independientemente del tamaño del dataset.

**Singleton ModelManager con fallback MLflow → joblib** — La API carga el modelo una sola vez al arranque. Intenta primero el MLflow Registry; si el servidor no está disponible (desarrollo local sin Docker), cae back a archivos `.joblib` locales. La protección anti-DNS rebinding de MLflow 2.x requiere `--allowed-hosts mlflow` cuando se llama desde dentro de la red Docker.

**Jenkins en lugar de GitHub Actions** — Todo es local. Los contenedores Jenkins comparten el socket Docker del host y pueden conectar directamente a Postgres y MLflow, algo que los runners hosteados de GitHub Actions no pueden hacer sin tunelado o infraestructura mock.

**Manejo del desbalance de clases** — El dataset tiene un 6.4% de tasa de churn. Los modelos se entrenaron con `scale_pos_weight` (XGBoost/LightGBM). El umbral óptimo para F1 (≈0.89) se calibra post-entrenamiento en lugar de fijarlo en 0.5.

---

## Descripción de los datos

| Archivo | Registros | Período | Descripción |
| --- | --- | --- | --- |
| `train.csv` | 992,931 | Feb 2017 | Etiquetas de entrenamiento |
| `train_v2.csv` | 970,960 | Mar 2017 | Etiquetas de validación |
| `sample_submission_v2.csv` | 907,471 | Abr 2017 | Set de test (sin etiquetas) |
| `transactions.csv` | 21.5M filas | 2015–2017 | Historial de pagos y planes |
| `user_logs.csv` | 392M filas (~28 GB) | 2015–2017 | Comportamiento de escucha diario |

Los datos crudos son de solo lectura en `data/raw/`. Todas las transformaciones producen outputs en `data/processed/`.
