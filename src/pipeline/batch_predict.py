"""
Fase 2, Tarea 9 — Batch prediction pipeline.

Carga el modelo Production desde MLflow Registry, lee las features
pre-computadas del Feature Store (features_monthly) y escribe las
predicciones en la tabla predictions.

Operación idempotente: si el período + model_version ya existe, lo reemplaza.

Uso:
  python -m src.pipeline.batch_predict --period 2017-04
  python -m src.pipeline.batch_predict --period 2017-04 --stage Staging
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.api.config import get_settings
from src.pipeline.db import get_engine, check_connection
from src.models.retrain_submit_14 import FEATURE_COLS_V5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 50_000


# ── CARGA DE MODELO ───────────────────────────────────────────────────────────

def load_model(model_name: str, stage: str):
    """Carga el modelo desde MLflow Registry en el stage indicado."""
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    model_uri = f"models:/{model_name}/{stage}"
    log.info(f"Cargando modelo: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # Obtener versión activa para el registro de predicciones
    client   = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=[stage])
    version  = versions[0].version if versions else "unknown"
    log.info(f"  Modelo {model_name} v{version} cargado")
    return model, version


# ── LECTURA DEL FEATURE STORE ─────────────────────────────────────────────────

def read_features(period: str) -> pd.DataFrame:
    """Lee features del Feature Store para el período dado."""
    engine = get_engine()
    query  = text(
        "SELECT * FROM features_monthly WHERE period = :period ORDER BY msno"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"period": period})

    log.info(f"  {len(df):,} usuarios cargados desde Feature Store (período {period})")
    return df


# ── PREDICCIÓN EN LOTES ───────────────────────────────────────────────────────

def predict_batches(model, df: pd.DataFrame) -> np.ndarray:
    """Corre el modelo en batches para no saturar memoria."""
    probas = []
    n      = len(df)

    for start in range(0, n, BATCH_SIZE):
        batch       = df.iloc[start : start + BATCH_SIZE]
        batch_feats = batch[FEATURE_COLS_V5].copy()
        proba       = model.predict(batch_feats)
        probas.append(proba)

        pct = min(start + BATCH_SIZE, n) / n * 100
        log.info(f"  Predicciones: {min(start + BATCH_SIZE, n):,}/{n:,} ({pct:.0f}%)")

    return np.concatenate(probas)


# ── ESCRITURA EN PREDICTIONS ──────────────────────────────────────────────────

def write_predictions(
    msno: pd.Series,
    probas: np.ndarray,
    period: str,
    model_version: str,
    source: str = "batch",
    chunksize: int = 10_000,
) -> int:
    """
    Escribe predicciones en la tabla predictions.
    Idempotente: elimina las del mismo período + versión antes de escribir.
    """
    engine = get_engine()
    now    = datetime.utcnow()

    out = pd.DataFrame({
        "msno":          msno.values,
        "churn_prob":    probas.astype(np.float32),
        "churn_label":   (probas >= 0.5).astype(int),
        "model_version": model_version,
        "period":        period,
        "source":        source,
        "predicted_at":  now,
    })

    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM predictions "
                "WHERE period = :period AND model_version = :version"
            ),
            {"period": period, "version": model_version},
        )

    out.to_sql(
        "predictions",
        engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    return len(out)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run(period: str, stage: str = "Production") -> None:
    cfg = get_settings()

    log.info("═" * 60)
    log.info(f"BATCH PREDICT — período {period} | stage {stage}")
    log.info("═" * 60)

    # ── Verificar DB ──────────────────────────────────────────────────────────
    if not check_connection():
        raise RuntimeError("Postgres no disponible — verifica que el contenedor esté corriendo")

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    try:
        model, model_version = load_model(cfg.mlflow_model_name, stage)
    except Exception as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo '{cfg.mlflow_model_name}/{stage}' "
            f"desde {cfg.mlflow_tracking_uri}. "
            f"¿Corriste train_mlflow_17.py primero?\nError: {e}"
        )

    # ── Leer features ─────────────────────────────────────────────────────────
    df = read_features(period)
    if df.empty:
        raise RuntimeError(
            f"Sin features para período '{period}'. "
            "Corre compute_features.py primero."
        )

    missing = [c for c in FEATURE_COLS_V5 if c not in df.columns]
    if missing:
        raise RuntimeError(f"Features faltantes en Feature Store: {missing}")

    # ── Predecir ──────────────────────────────────────────────────────────────
    log.info(f"Prediciendo {len(df):,} usuarios en batches de {BATCH_SIZE:,}...")
    probas = predict_batches(model, df)

    log.info("Distribución de predicciones:")
    log.info(f"  mean  : {probas.mean():.4f}")
    log.info(f"  p50   : {np.median(probas):.4f}")
    log.info(f"  p90   : {np.percentile(probas, 90):.4f}")
    log.info(f"  p99   : {np.percentile(probas, 99):.4f}")
    log.info(f"  churn_label=1: {(probas >= 0.5).mean():.2%}")

    # ── Escribir predicciones ─────────────────────────────────────────────────
    log.info(f"Escribiendo predicciones en DB (período {period}, v{model_version})...")
    n = write_predictions(df["msno"], probas, period, model_version)
    log.info(f"  ✓ {n:,} predicciones escritas")

    log.info("Batch predict completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch churn prediction")
    parser.add_argument("--period", required=True, help="Período YYYY-MM (ej: 2017-04)")
    parser.add_argument("--stage",  default="Production", help="MLflow stage (default: Production)")
    args = parser.parse_args()
    run(args.period, args.stage)
