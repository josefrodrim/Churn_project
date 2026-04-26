"""
Fase 2, Tarea 8 — Batch feature computation pipeline.

Lee datos crudos, computa las 48 features v5 para todos los usuarios
activos de un período, valida rangos y escribe en features_monthly.

Operación idempotente: si el período ya existe en la tabla, lo reemplaza.

Uso:
  python -m src.pipeline.compute_features --period 2017-03
  python -m src.pipeline.compute_features --period 2017-04
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.api.config import get_settings
from src.pipeline.db import get_engine, check_connection
from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import (
    load_log_features,
    MAX_DATE_MAR, RECENT_CUTOFF_MAR, EXTRA_CUTOFFS_MAR_V2,
)
from src.models.train_06 import RANDOM_STATE
from src.models.retrain_submit_13 import (
    build_member_features, build_expiry_features, build_churn_lag,
)
from src.models.retrain_submit_14 import (
    FEATURE_COLS_V5,
    build_tx_recency_features,
    join_all_v5,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_RAW = ROOT / "data" / "raw"

# ── CONFIGURACIÓN POR PERÍODO ─────────────────────────────────────────────────
# Mapea YYYY-MM → (prediction_date, log_cutoff, label_file)
# prediction_date: fecha de referencia para tenure_days y days_until_expire
# log_cutoff:      último día de logs disponibles para ese período

PERIOD_CONFIG = {
    "2017-02": {
        "prediction_date": pd.Timestamp("2017-02-28"),
        "log_cutoff":      pd.Timestamp("2017-02-28"),
        "label_file":      "train.csv",
        "log_kwargs":      {},  # usa cache default (feb)
    },
    "2017-03": {
        "prediction_date": pd.Timestamp("2017-03-31"),
        "log_cutoff":      pd.Timestamp("2017-03-31"),
        "label_file":      "train_v2.csv",
        "log_kwargs": {
            "max_date":      MAX_DATE_MAR,
            "recent_cutoff": RECENT_CUTOFF_MAR,
            "extra_cutoffs": EXTRA_CUTOFFS_MAR_V2,
        },
    },
    "2017-04": {
        "prediction_date": pd.Timestamp("2017-04-30"),
        "log_cutoff":      pd.Timestamp("2017-03-31"),   # sin logs de abril
        "label_file":      "sample_submission_v2.csv",
        "log_kwargs": {
            "max_date":      MAX_DATE_MAR,
            "recent_cutoff": RECENT_CUTOFF_MAR,
            "extra_cutoffs": EXTRA_CUTOFFS_MAR_V2,
        },
    },
}


# ── VALIDACIÓN ────────────────────────────────────────────────────────────────

RANGE_CHECKS = {
    "age":              (0, 120),
    "completion_ratio": (0, 1),
    "avg_discount_pct": (0, 1),
    "churn_prob":       (0, 1),   # solo en predictions, no en features
}

BINARY_COLS = [
    "ever_canceled", "last_is_cancel", "last_is_auto_renew",
    "bd_valid", "has_member_record", "has_log_record",
    "is_expired", "auto_renew_at_expire", "cancel_at_expire",
    "prev_churn", "cancel_before_expire", "had_tx_last_7d", "had_tx_last_30d",
]


def validate_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa registros válidos de inválidos.
    Devuelve (valid_df, invalid_df).
    """
    mask = pd.Series(True, index=df.index)

    # Sin nulos en las features
    null_mask = df[FEATURE_COLS_V5].isnull().any(axis=1)
    mask &= ~null_mask

    # Rangos numéricos
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col in df.columns:
            mask &= df[col].between(lo, hi)

    # Columnas binarias en {0, 1}
    for col in BINARY_COLS:
        if col in df.columns:
            mask &= df[col].isin([0, 1])

    valid   = df[mask].copy()
    invalid = df[~mask].copy()
    return valid, invalid


# ── ESCRITURA EN POSTGRES ────────────────────────────────────────────────────

def write_features(df: pd.DataFrame, period: str, chunksize: int = 10_000) -> int:
    """
    Escribe features en features_monthly.
    Elimina el período si ya existe (idempotente).
    Devuelve número de filas escritas.
    """
    engine = get_engine()
    cols   = ["msno", "period", "computed_at"] + FEATURE_COLS_V5

    out = df[["msno"] + FEATURE_COLS_V5].copy()
    out["period"]      = period
    out["computed_at"] = datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM features_monthly WHERE period = :period"),
            {"period": period},
        )

    out[cols].to_sql(
        "features_monthly",
        engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    return len(out)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def run(period: str) -> None:
    if period not in PERIOD_CONFIG:
        raise ValueError(
            f"Período '{period}' no soportado. Opciones: {list(PERIOD_CONFIG)}"
        )

    cfg_period = PERIOD_CONFIG[period]
    prediction_date = cfg_period["prediction_date"]
    log_cutoff      = cfg_period["log_cutoff"]
    label_file      = cfg_period["label_file"]
    log_kwargs      = cfg_period["log_kwargs"]

    log.info("═" * 60)
    log.info(f"COMPUTE FEATURES — período {period}")
    log.info(f"  prediction_date : {prediction_date.date()}")
    log.info(f"  log_cutoff      : {log_cutoff.date()}")
    log.info(f"  usuarios desde  : {label_file}")
    log.info("═" * 60)

    # ── Verificar conexión DB ─────────────────────────────────────────────────
    if not check_connection():
        log.warning("Postgres no disponible — se omite escritura en DB")
        db_available = False
    else:
        log.info("Postgres conectado")
        db_available = True

    # ── Cargar datos crudos ───────────────────────────────────────────────────
    log.info("Cargando logs de usuario...")
    raw_agg = load_log_features(**log_kwargs) if log_kwargs else load_log_features()
    log.info(f"  {len(raw_agg):,} usuarios en log cache")

    log.info("Cargando transacciones...")
    tx         = load_transactions()
    tx_feats   = build_user_features(tx)
    tx_recency = build_tx_recency_features(tx)
    log.info(f"  {len(tx_feats):,} usuarios con transacciones")

    # ── Construir features ────────────────────────────────────────────────────
    log.info(f"Construyendo features v5 ({len(FEATURE_COLS_V5)} features)...")
    labels      = pd.read_csv(DATA_RAW / label_file)[["msno"]]
    member_feat = build_member_features(prediction_date)
    expiry_feat = build_expiry_features(tx, log_cutoff)
    churn_lag   = build_churn_lag(prediction_date)

    df = join_all_v5(
        labels, tx_feats, member_feat, raw_agg,
        expiry_feat, churn_lag, tx_recency,
    )
    log.info(f"  {len(df):,} usuarios | nulos antes de validar: {df[FEATURE_COLS_V5].isnull().sum().sum()}")

    # ── Validar ───────────────────────────────────────────────────────────────
    valid, invalid = validate_features(df)
    log.info(f"  Válidos: {len(valid):,} | Inválidos: {len(invalid):,} ({len(invalid)/len(df):.2%})")

    if len(invalid) > 0:
        top_nulls = df[FEATURE_COLS_V5].isnull().sum().nlargest(5)
        log.warning(f"  Top features con nulos:\n{top_nulls[top_nulls > 0]}")

    # ── Estadísticas de distribución ──────────────────────────────────────────
    log.info("Distribución de features clave:")
    for col in ["days_until_expire", "days_since_last", "n_renewals", "prev_churn"]:
        if col in valid.columns:
            log.info(
                f"  {col:30s}: mean={valid[col].mean():.2f} "
                f"std={valid[col].std():.2f} "
                f"p50={valid[col].median():.2f}"
            )

    # ── Escribir en DB ────────────────────────────────────────────────────────
    if db_available:
        log.info(f"Escribiendo {len(valid):,} registros en features_monthly...")
        n_written = write_features(valid, period)
        log.info(f"  ✓ {n_written:,} filas escritas para período {period}")
    else:
        log.info("(Escritura en DB omitida — Postgres no disponible)")
        log.info(f"  Shape del DataFrame: {valid.shape}")

    log.info("Compute features completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch feature computation")
    parser.add_argument(
        "--period",
        required=True,
        help="Período a computar, formato YYYY-MM (ej: 2017-03)",
    )
    args = parser.parse_args()
    run(args.period)
