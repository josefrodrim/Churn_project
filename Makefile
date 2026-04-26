.PHONY: up down build restart logs test test-local \
        retrain compute-features predict drift promote shell ps help

# ── Variables ─────────────────────────────────────────────────────────────────
PERIOD ?= $(shell date +%Y-%m)
API    := docker compose exec -T api
PYTHON := python -m

# ── Stack ────────────────────────────────────────────────────────────────────
up:                       ## Levanta todos los servicios en background
	docker compose up --build -d
	@echo ""
	@echo "  Stack levantado:"
	@echo "    API      →  http://localhost:8000/docs"
	@echo "    MLflow   →  http://localhost:5000"
	@echo "    Jenkins  →  http://localhost:8080"
	@echo "    Grafana  →  http://localhost:3000"
	@echo "    Postgres →  localhost:5432"

down:                     ## Para y elimina todos los contenedores
	docker compose down

build:                    ## Rebuild de imágenes sin cache
	docker compose build --no-cache

restart:                  ## Reinicia el servicio api sin reconstruir
	docker compose restart api

ps:                       ## Estado de los contenedores
	docker compose ps

logs:                     ## Logs en vivo del API
	docker compose logs -f api

logs-all:                 ## Logs en vivo de todos los servicios
	docker compose logs -f

# ── Tests ────────────────────────────────────────────────────────────────────
test:                     ## Corre pytest dentro del contenedor api
	$(API) pytest tests/ -v --tb=short

test-local:               ## Corre pytest localmente (sin Docker)
	pytest tests/ -v --tb=short

# ── ML Pipeline ──────────────────────────────────────────────────────────────
compute-features:         ## Computa features para el período (default: mes actual)
	@echo "Computando features para período $(PERIOD)..."
	$(API) $(PYTHON) src.pipeline.compute_features --period $(PERIOD)

retrain:                  ## Entrena modelos y registra en MLflow
	@echo "Reentrenando ensemble..."
	$(API) $(PYTHON) src.models.train_mlflow_17

predict:                  ## Batch prediction para el período (default: mes actual)
	@echo "Prediciendo churn para período $(PERIOD)..."
	$(API) $(PYTHON) src.pipeline.batch_predict --period $(PERIOD)

pipeline:                 ## Pipeline completo: features → retrain → predict
	$(MAKE) compute-features PERIOD=$(PERIOD)
	$(MAKE) retrain
	$(MAKE) predict PERIOD=$(PERIOD)

# ── Monitoreo ─────────────────────────────────────────────────────────────────
drift:                    ## Genera reporte de data drift con Evidently
	$(API) $(PYTHON) src.monitoring.drift

# ── Model Registry ────────────────────────────────────────────────────────────
promote:                  ## Promueve modelo de Staging a Production en MLflow
	@echo "Promoviendo modelo Staging → Production..."
	$(API) python -c "\
import mlflow; \
mlflow.set_tracking_uri('http://mlflow:5000'); \
c = mlflow.tracking.MlflowClient(); \
vs = c.get_latest_versions('churn-ensemble', stages=['Staging']); \
v = vs[0] if vs else None; \
print(f'Promoviendo v{v.version}...') if v else print('Sin modelos en Staging'); \
c.transition_model_version_stage('churn-ensemble', v.version, 'Production') if v else None; \
print('Listo.') if v else None"

# ── Utilidades ────────────────────────────────────────────────────────────────
shell:                    ## Shell interactivo dentro del contenedor api
	docker compose exec api /bin/bash

shell-db:                 ## psql directo a Postgres
	docker compose exec postgres psql -U $${POSTGRES_USER:-churn_user} -d $${POSTGRES_DB:-churn}

help:                     ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
