"""
Rutas de predicción: individual, batch y consulta de predicciones almacenadas.
Todas requieren X-API-Key.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.config import Settings, get_settings
from src.api.dependencies import get_db, log_prediction, model_manager, verify_api_key
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.models.retrain_submit_14 import FEATURE_COLS_V5

log    = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


def _feature_record_to_df(features) -> pd.DataFrame:
    """Convierte un FeatureRecord a DataFrame de una fila con las columnas correctas."""
    row = {k: getattr(features, k) for k in FEATURE_COLS_V5}
    return pd.DataFrame([row])[FEATURE_COLS_V5]


def _build_response(
    msno: str,
    prob: float,
    cfg: Settings,
) -> PredictionResponse:
    return PredictionResponse(
        msno          = msno,
        churn_prob    = round(float(prob), 6),
        churn_label   = int(prob >= 0.5),
        model_version = model_manager.version,
        predicted_at  = datetime.utcnow(),
    )


# ── POST /predict — predicción individual ────────────────────────────────────

@router.post("", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    cfg:     Settings      = Depends(get_settings),
    db:      AsyncSession  = Depends(get_db),
    _:       str           = Depends(verify_api_key),
):
    """
    Predicción en tiempo real para un usuario.
    Acepta el feature vector completo y devuelve la probabilidad de churn.
    La predicción queda loggeada en la tabla predictions.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible",
        )

    df   = _feature_record_to_df(request.features)
    prob = float(model_manager.predict(df)[0])

    # Log asíncrono — no bloquea la respuesta
    period = datetime.utcnow().strftime("%Y-%m")
    await log_prediction(db, request.features.msno, prob, int(prob >= 0.5), period)

    return _build_response(request.features.msno, prob, cfg)


# ── POST /predict/batch — predicción en lote ─────────────────────────────────

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    cfg:     Settings     = Depends(get_settings),
    db:      AsyncSession = Depends(get_db),
    _:       str          = Depends(verify_api_key),
):
    """
    Predicción para un lote de hasta 10,000 usuarios en una sola llamada.
    Más eficiente que N llamadas individuales para casos de uso batch.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible",
        )

    rows   = [
        {k: getattr(u, k) for k in FEATURE_COLS_V5}
        for u in request.users
    ]
    df     = pd.DataFrame(rows)[FEATURE_COLS_V5]
    probas = model_manager.predict(df)
    now    = datetime.utcnow()
    period = now.strftime("%Y-%m")

    responses = []
    for user, prob in zip(request.users, probas):
        prob = float(prob)
        responses.append(
            PredictionResponse(
                msno          = user.msno,
                churn_prob    = round(prob, 6),
                churn_label   = int(prob >= 0.5),
                model_version = model_manager.version,
                predicted_at  = now,
            )
        )
        await log_prediction(db, user.msno, prob, int(prob >= 0.5), period, source="batch")

    return BatchPredictionResponse(
        predictions   = responses,
        model_version = model_manager.version,
        total         = len(responses),
        predicted_at  = now,
    )


# ── GET /predict/{msno} — consulta predicción pre-computada ──────────────────

@router.get("/{msno}", response_model=PredictionResponse)
async def get_prediction(
    msno:   str,
    period: str | None = None,
    db:     AsyncSession = Depends(get_db),
    _:      str          = Depends(verify_api_key),
):
    """
    Devuelve la predicción más reciente de un usuario desde el Feature Store.
    Más rápido que /predict para consultas de usuarios ya procesados en batch.

    Parámetro opcional `period` (YYYY-MM) para consultar un período específico.
    """
    if period:
        query  = text(
            "SELECT * FROM predictions WHERE msno = :msno AND period = :period "
            "ORDER BY predicted_at DESC LIMIT 1"
        )
        params = {"msno": msno, "period": period}
    else:
        query  = text(
            "SELECT * FROM predictions WHERE msno = :msno "
            "ORDER BY predicted_at DESC LIMIT 1"
        )
        params = {"msno": msno}

    result = await db.execute(query, params)
    row    = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sin predicción para msno='{msno}'" + (f" período={period}" if period else ""),
        )

    return PredictionResponse(
        msno          = row["msno"],
        churn_prob    = row["churn_prob"],
        churn_label   = row["churn_label"],
        model_version = row["model_version"],
        predicted_at  = row["predicted_at"],
    )
