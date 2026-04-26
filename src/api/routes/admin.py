"""
Rutas de administración: health check y model info.
No requieren API key — son públicas para load balancers y dashboards.
"""

import logging
from datetime import datetime

import mlflow
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.config import Settings, get_settings
from src.api.dependencies import get_db, model_manager, verify_api_key
from src.api.schemas import HealthResponse, ModelInfo
from fastapi import Security
from src.models.retrain_submit_14 import FEATURE_COLS_V5

log    = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])


@router.get("/health", response_model=HealthResponse)
async def health(
    db: AsyncSession = Depends(get_db),
    cfg: Settings    = Depends(get_settings),
):
    """Verifica que el API está vivo, el modelo cargado y la DB accesible."""
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status        = "ok" if (model_manager.is_loaded and db_ok) else "degraded",
        model_loaded  = model_manager.is_loaded,
        db_connected  = db_ok,
        model_version = model_manager.version,
    )


@router.get("/model/info", response_model=ModelInfo)
async def model_info(cfg: Settings = Depends(get_settings)):
    """Devuelve metadata del modelo activo en producción."""
    log_loss_val = None
    roc_auc_val  = None
    registered   = None

    try:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        client   = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(
            cfg.mlflow_model_name, stages=[cfg.mlflow_model_stage]
        )
        if versions:
            mv      = versions[0]
            run     = client.get_run(mv.run_id)
            metrics = run.data.metrics
            log_loss_val = metrics.get("ensemble_val_logloss")
            roc_auc_val  = metrics.get("ensemble_val_auc")
            registered   = datetime.fromtimestamp(mv.creation_timestamp / 1000)
    except Exception as e:
        log.warning(f"No se pudo consultar MLflow: {e}")

    return ModelInfo(
        name          = cfg.mlflow_model_name,
        version       = model_manager.version,
        stage         = cfg.mlflow_model_stage,
        feature_count = len(FEATURE_COLS_V5),
        log_loss_val  = log_loss_val,
        roc_auc_val   = roc_auc_val,
        registered_at = registered,
    )


@router.get("/predictions/recent")
async def predictions_recent(
    limit: int = 100,
    db: AsyncSession  = Depends(get_db),
    _: str            = Depends(verify_api_key),
):
    """Últimas N predicciones — usado por el dashboard del frontend."""
    result = await db.execute(
        text("""
            SELECT msno, churn_prob, churn_label, model_version, source, predicted_at
            FROM predictions
            ORDER BY predicted_at DESC
            LIMIT :limit
        """),
        {"limit": min(limit, 500)},
    )
    rows = result.mappings().all()
    return {"predictions": [dict(r) for r in rows], "count": len(rows)}
