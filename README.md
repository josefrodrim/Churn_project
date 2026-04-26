# KKBox Churn Prediction — MLOps Pipeline

> **ES** | [EN](#english-version)

---

## Versión en Español

Proyecto end-to-end de predicción de churn para la competencia [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge). Incluye exploración de datos, ingeniería de features, entrenamiento de modelos, y un pipeline completo de MLOps para llevarlo a producción local.

### Resultados del modelo

| Submission | Score público (log loss) | Descripción |
| --- | --- | --- |
| v1 LightGBM tuneado | 0.37856 | Baseline — 30 features |
| v3 Fix temporal | 0.30398 | Fix de offset en `days_since_last` |
| v4 Expiry features | 0.23528 | +6 features de expiración de membresía |
| v5 raw (mejor single) | **0.23504** | 48 features, sin calibración |
| v10 Blend 3 modelos | **0.23412** | LightGBM + XGBoost + CatBoost |

### Stack tecnológico

```text
Entrenamiento    → LightGBM + XGBoost + CatBoost (blend ensemble)
Tracking         → MLflow (experimentos + model registry)
API              → FastAPI
Base de datos    → PostgreSQL
Contenedores     → Docker + Docker Compose
CI/CD            → Jenkins (100% local)
Monitoreo        → Evidently AI + Grafana + Prometheus
```

### Levantar el stack completo

```bash
cp .env.example .env          # configurar variables de entorno
docker compose up --build     # levanta los 5 servicios
```

| Servicio | URL |
| --- | --- |
| API REST | <http://localhost:8000> |
| MLflow UI | <http://localhost:5000> |
| Jenkins | <http://localhost:8080> |
| Grafana | <http://localhost:3000> |
| PostgreSQL | localhost:5432 |

### Estructura del proyecto

```text
src/
  api/          → FastAPI service (schemas, config, routes)
  eda/          → Análisis exploratorio y carga de datos
  features/     → Feature engineering
  models/       → Scripts de entrenamiento (numbered: 06–17)
  pipeline/     → Batch feature computation + batch prediction
  monitoring/   → Drift detection, performance tracking, alertas
infra/
  docker/       → Dockerfiles
  postgres/     → SQL de inicialización del schema
  jenkins/      → Jenkinsfiles para los 4 pipelines
  grafana/      → Dashboards y datasources
data/
  raw/          → Datos originales (DVC tracked, read-only)
  processed/    → Features pre-computadas (DVC tracked)
models/         → Modelos serializados (joblib/cbm)
submissions/    → CSVs subidos a Kaggle con historial de scores
tests/
  unit/         → Tests de feature engineering
  integration/  → Tests del API
notebooks/      → Exploración y experimentos
reports/
  figures/      → Gráficas generadas
  monitoring/   → Reportes de drift (Evidently HTML)
```

### Fases del pipeline MLOps

| Fase | Estado | Descripción |
| --- | --- | --- |
| 0 — Contratos | ✅ | Schemas Pydantic, tablas Postgres, secrets management |
| 1 — MLflow | ✅ | Tracking de experimentos + model registry |
| 2 — Feature Store | ✅ | Pipeline batch de features + validación |
| 3 — FastAPI | ✅ | API de predicción (individual + batch) |
| 4 — Docker | ✅ | Dockerfiles + docker-compose.yml |
| 5 — Tests | ⬜ | Unit, integration, quality gate |
| 6 — Jenkins | ⬜ | CI, CD, retrain mensual, monitoring diario |
| 7 — Monitoreo | ⬜ | Evidently + Grafana + alertas automáticas |

### Comandos útiles

```bash
make up          # Levanta todo el stack
make down        # Baja el stack
make test        # Corre pytest
make retrain     # Reentrenamiento completo + registro en MLflow
make predict     # Batch prediction mensual
make drift       # Genera reporte de drift con Evidently
make promote     # Promueve modelo de Staging a Production
make logs        # Logs del API en tiempo real
```

---

## English Version

End-to-end churn prediction project for the [WSDM KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge). Covers data exploration, feature engineering, model training, and a complete local MLOps pipeline.

### Model Results

| Submission | Public Score (log loss) | Description |
| --- | --- | --- |
| v1 Tuned LightGBM | 0.37856 | Baseline — 30 features |
| v3 Temporal fix | 0.30398 | Fixed `days_since_last` date offset |
| v4 Expiry features | 0.23528 | +6 membership expiry features |
| v5 raw (best single) | **0.23504** | 48 features, no calibration |
| v10 3-model blend | **0.23412** | LightGBM + XGBoost + CatBoost |

### Tech Stack

```text
Training         → LightGBM + XGBoost + CatBoost (blend ensemble)
Experiment Track → MLflow (experiments + model registry)
API              → FastAPI
Database         → PostgreSQL
Containers       → Docker + Docker Compose
CI/CD            → Jenkins (100% local)
Monitoring       → Evidently AI + Grafana + Prometheus
```

### Start the Full Stack

```bash
cp .env.example .env          # set environment variables
docker compose up --build     # starts all 5 services
```

| Service | URL |
| --- | --- |
| REST API | <http://localhost:8000> |
| MLflow UI | <http://localhost:5000> |
| Jenkins | <http://localhost:8080> |
| Grafana | <http://localhost:3000> |
| PostgreSQL | localhost:5432 |

### Project Structure

```text
src/
  api/          → FastAPI service (schemas, config, routes)
  eda/          → Exploratory analysis and data loading
  features/     → Feature engineering
  models/       → Training scripts (numbered: 06–17)
  pipeline/     → Batch feature computation + batch prediction
  monitoring/   → Drift detection, performance tracking, alerts
infra/
  docker/       → Dockerfiles
  postgres/     → Schema initialization SQL
  jenkins/      → Jenkinsfiles for the 4 pipelines
  grafana/      → Dashboards and datasources
data/
  raw/          → Original data (DVC tracked, read-only)
  processed/    → Pre-computed features (DVC tracked)
models/         → Serialized models (joblib/cbm)
submissions/    → Kaggle submission CSVs with score history
tests/
  unit/         → Feature engineering tests
  integration/  → API endpoint tests
notebooks/      → Exploration and experiments
reports/
  figures/      → Generated plots
  monitoring/   → Drift reports (Evidently HTML)
```

### MLOps Pipeline Phases

| Phase | Status | Description |
| --- | --- | --- |
| 0 — Contracts | ✅ | Pydantic schemas, Postgres tables, secrets management |
| 1 — MLflow | ✅ | Experiment tracking + model registry |
| 2 — Feature Store | ⬜ | Batch feature pipeline + validation |
| 3 — FastAPI | ⬜ | Prediction API (single + batch) |
| 4 — Docker | ✅ | Dockerfiles + docker-compose.yml |
| 5 — Tests | ⬜ | Unit, integration, quality gate |
| 6 — Jenkins | ⬜ | CI, CD, monthly retrain, daily monitoring |
| 7 — Monitoring | ⬜ | Evidently + Grafana + automatic alerts |

### Key Lessons Learned

- `days_until_expire` is the strongest churn signal — expired membership = almost certain churn
- Date offset between training and submission (reference date mismatch) causes severe calibration errors
- Isotonic calibration overfits badly on small holdout sets with imbalanced classes
- Proper temporal split: train(Feb) → validate(Mar) → predict(Apr)
- A 3-model blend (LightGBM + XGBoost + CatBoost) gives marginal improvement over the best single model
- XGBoost tuned on a small subsample underpredicts — always tune on representative data

### Data Overview

| File | Records | Period | Description |
| --- | --- | --- | --- |
| `train.csv` | 992,931 | Feb 2017 | Training labels |
| `train_v2.csv` | 970,960 | Mar 2017 | Validation labels |
| `sample_submission_v2.csv` | 907,471 | Apr 2017 | Test set (no labels) |
| `transactions.csv` | 21.5M | Jan 2015 – Feb 2017 | Payment history |
| `user_logs.csv` | 392M | Jan 2015 – Feb 2017 | Daily listening behavior (~28 GB) |
