"""
Dependencias de FastAPI: modelo, DB y autenticación.

- ModelManager  : carga el ensemble una vez al arrancar (MLflow → joblib fallback)
- get_db()      : sesión async de Postgres por request
- verify_api_key: valida X-API-Key header
"""

import logging
from datetime import datetime
from typing import AsyncGenerator

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.api.config import Settings, get_settings
from src.models.retrain_submit_14 import FEATURE_COLS_V5
from src.models.ensemble_16 import CATEG_COLS, to_catboost_df

log = logging.getLogger(__name__)

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── MODEL MANAGER ─────────────────────────────────────────────────────────────

class ModelManager:
    """
    Singleton que mantiene el ensemble en memoria durante la vida del proceso.

    Estrategia de carga:
      1. MLflow Registry (Production stage) — cuando el server está disponible
      2. Fallback a joblib — para desarrollo local sin docker compose
    """

    def __init__(self):
        self._model       = None
        self._version     = "unknown"
        self._loaded      = False
        self._loaded_from = "none"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    @property
    def source(self) -> str:
        return self._loaded_from

    def load(self, cfg: Settings) -> None:
        """Intenta MLflow; si falla, carga los joblib guardados localmente."""
        try:
            self._load_from_mlflow(cfg)
        except Exception as e:
            log.warning(f"MLflow no disponible ({e.__class__.__name__}: {e})")
            log.warning("Cargando modelo desde joblib (fallback local)...")
            self._load_from_joblib(cfg)

    def _load_from_mlflow(self, cfg: Settings) -> None:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        uri = f"models:/{cfg.mlflow_model_name}/{cfg.mlflow_model_stage}"
        self._model = mlflow.pyfunc.load_model(uri)

        client   = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            cfg.mlflow_model_name, stages=[cfg.mlflow_model_stage]
        )
        self._version     = versions[0].version if versions else "mlflow-unknown"
        self._loaded      = True
        self._loaded_from = "mlflow"
        log.info(f"Modelo cargado desde MLflow: {uri} (v{self._version})")

    def _load_from_joblib(self, cfg: Settings) -> None:
        import joblib
        from pathlib import Path
        from catboost import CatBoostClassifier
        from src.models.train_mlflow_17 import ChurnEnsemble

        root     = Path(__file__).resolve().parents[2]
        models   = root / "models"

        lgbm = joblib.load(models / "lgbm_ensemble_v1.joblib")["model"]
        xgb  = joblib.load(models / "xgb_v1.joblib")["model"]
        cat  = CatBoostClassifier()
        cat.load_model(str(models / "catboost_v1.cbm"))

        ensemble      = ChurnEnsemble(lgbm, xgb, cat, FEATURE_COLS_V5, CATEG_COLS)
        self._model   = ensemble          # tiene .predict(context, df)
        self._version = "local-joblib"
        self._loaded  = True
        self._loaded_from = "joblib"
        log.info("Modelo cargado desde joblib (fallback)")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Modelo no cargado")
        if self._loaded_from == "joblib":
            return self._model.predict(None, df)
        return self._model.predict(df)


# Instancia global — se llena en el lifespan de app.py
model_manager = ModelManager()


# ── DB ASYNC ──────────────────────────────────────────────────────────────────

_async_engine: AsyncEngine | None = None
_AsyncSessionLocal = None


def init_db(cfg: Settings) -> None:
    global _async_engine, _AsyncSessionLocal
    url = (
        f"postgresql+asyncpg://{cfg.postgres_user}:{cfg.postgres_password}"
        f"@{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"
    )
    _async_engine      = create_async_engine(url, pool_pre_ping=True, pool_size=10)
    _AsyncSessionLocal = sessionmaker(
        _async_engine, class_=AsyncSession, expire_on_commit=False
    )
    log.info(f"Async DB engine creado → {cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}")


async def close_db() -> None:
    if _async_engine:
        await _async_engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    if _AsyncSessionLocal is None:
        raise RuntimeError("DB no inicializada — llama init_db() al arrancar")
    async with _AsyncSessionLocal() as session:
        yield session


# ── PREDICTION LOGGER ─────────────────────────────────────────────────────────

async def log_prediction(
    session: AsyncSession,
    msno: str,
    churn_prob: float,
    churn_label: int,
    period: str,
    source: str = "api",
) -> None:
    """Persiste una predicción en la tabla predictions (fire-and-forget seguro)."""
    try:
        from sqlalchemy import text
        await session.execute(
            text(
                "INSERT INTO predictions "
                "(msno, churn_prob, churn_label, model_version, period, source, predicted_at) "
                "VALUES (:msno, :prob, :label, :version, :period, :source, :ts)"
            ),
            {
                "msno":    msno,
                "prob":    float(churn_prob),
                "label":   int(churn_label),
                "version": model_manager.version,
                "period":  period,
                "source":  source,
                "ts":      datetime.utcnow(),
            },
        )
        await session.commit()
    except Exception as e:
        log.warning(f"No se pudo loggear predicción para {msno}: {e}")


# ── AUTH ──────────────────────────────────────────────────────────────────────

async def verify_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
    cfg: Settings = Depends(get_settings),
) -> str:
    if api_key is None or api_key != cfg.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key inválida o ausente",
        )
    return api_key
