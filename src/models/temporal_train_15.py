"""
Split temporal correcto para KKBox Churn.

Estructura temporal del concurso:
  ENTRENAR  → train.csv       (feb 2017, LOG_CUTOFF=2017-02-28)
  VALIDAR   → train_v2.csv    (mar 2017, LOG_CUTOFF=2017-03-31)  ← métricas honestas
  PREDECIR  → submission_v2   (abr 2017, LOG_CUTOFF=2017-03-31)

Cada período usa su propio LOG_CUTOFF como referencia para days_since_last y
days_until_expire — sin offset sistemático entre entrenamiento y validación.

Modelo final: reentrenado en feb+mar combinados antes de predecir abril.
Salida: submissions/submission_v6_temporal.csv
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_RAW        = ROOT / "data" / "raw"
DATA_PROC       = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

# Cada período usa su propia fecha de corte
CUTOFF_FEB = pd.Timestamp("2017-02-28")
CUTOFF_MAR = pd.Timestamp("2017-03-31")
CUTOFF_APR = pd.Timestamp("2017-04-30")  # solo para tenure_days en submission

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import (
    load_log_features, build_log_features,
    MAX_DATE, RECENT_CUTOFF,          # feb defaults
    MAX_DATE_MAR, RECENT_CUTOFF_MAR,  # mar
)
from src.features.build_features_05 import impute_missing
from src.models.train_06 import TARGET, RANDOM_STATE
from src.models.retrain_submit_13 import build_expiry_features, build_churn_lag

# 36 features (v4) — consistentes para feb, mar y submission
FEATURE_COLS = [
    # transacciones base
    "n_transactions", "n_cancels", "ever_canceled",
    "avg_discount_pct", "avg_plan_days", "avg_price",
    "n_unique_plans", "n_payment_methods",
    "last_is_cancel", "last_is_auto_renew",
    "last_plan_days", "last_price", "last_list_price",
    "price_trend", "last_payment_method",
    # members
    "city", "registered_via", "gender_enc",
    "age", "bd_valid", "tenure_days", "has_member_record",
    # user logs
    "n_days", "avg_daily_secs", "avg_daily_completed",
    "avg_daily_unq", "completion_ratio",
    "days_since_last", "listening_trend", "has_log_record",
    # expiry + lag (v4)
    "days_until_expire", "is_expired",
    "auto_renew_at_expire", "cancel_at_expire",
    "n_renewals", "prev_churn",
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def build_member_features(cutoff: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "members_v3.csv")
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["gender"] = df["gender"].fillna("unknown")
    df["gender_enc"] = df["gender"].map({"male": 0, "female": 1, "unknown": 2})
    df["bd_valid"] = df["bd"].between(1, 99).astype(int)
    df["age"] = np.where(df["bd_valid"], df["bd"], np.nan)
    df["tenure_days"] = (cutoff - df["registration_init_time"]).dt.days
    return df[["msno", "city", "registered_via", "gender_enc", "age", "bd_valid", "tenure_days"]]


def build_period_features(
    labels_df: pd.DataFrame,
    tx_feats: pd.DataFrame,
    log_cache: pd.DataFrame,
    expiry_feats: pd.DataFrame,
    churn_lag: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Construye features para un período dado usando su propio cutoff como referencia."""
    member_feats = build_member_features(cutoff)

    log_feats = build_log_features(
        log_cache,
        prediction_date=cutoff,
        days_since_ref=cutoff,  # referencia fija = cutoff del período
    )

    df = labels_df.copy()
    df = df.merge(tx_feats, on="msno", how="left")
    df = df.merge(member_feats, on="msno", how="left")
    df = df.merge(
        log_feats[[
            "msno", "n_days", "avg_daily_secs", "avg_daily_completed",
            "avg_daily_unq", "completion_ratio", "days_since_last", "listening_trend",
        ]],
        on="msno", how="left",
    )
    df["has_member_record"] = df["city"].notna().astype(int)
    df["has_log_record"]    = df["n_days"].notna().astype(int)
    df = impute_missing(df)

    df = df.merge(expiry_feats, on="msno", how="left")
    df = df.merge(churn_lag,    on="msno", how="left")

    df["days_until_expire"]    = df["days_until_expire"].fillna(df["days_until_expire"].median())
    df["is_expired"]           = df["is_expired"].fillna(1).astype(int)
    df["auto_renew_at_expire"] = df["auto_renew_at_expire"].fillna(0).astype(int)
    df["cancel_at_expire"]     = df["cancel_at_expire"].fillna(0).astype(int)
    df["n_renewals"]           = df["n_renewals"].fillna(0)
    df["prev_churn"]           = df["prev_churn"].fillna(0).astype(int)

    return df


def train_lgbm(X, y, best_params):
    model = LGBMClassifier(**best_params, is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1)
    model.fit(X, y)
    return model


def print_metrics(label, y_true, y_proba):
    print(f"  {label}")
    print(f"    ROC-AUC : {roc_auc_score(y_true, y_proba):.5f}")
    print(f"    PR-AUC  : {average_precision_score(y_true, y_proba):.5f}")
    print(f"    LogLoss : {log_loss(y_true, y_proba):.5f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("SPLIT TEMPORAL CORRECTO — feb → mar → abr")
    print("═" * 60)

    print("\n── Cargando datos compartidos ──")
    tx = load_transactions()
    tx_feats = build_user_features(tx)
    print(f"  {len(tx_feats):,} usuarios con features de transacciones")

    log_feb = load_log_features()                                           # hasta Feb 28
    log_mar = load_log_features(max_date=MAX_DATE_MAR, recent_cutoff=RECENT_CUTOFF_MAR)  # hasta Mar 31
    print(f"  log_feb: {len(log_feb):,} usuarios | log_mar: {len(log_mar):,} usuarios")

    best_params = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")["best_params"]

    # ── PERÍODO FEBRERO ───────────────────────────────────────────────────────
    print("\n── Período FEBRERO (train.csv) ──")
    labels_feb   = pd.read_csv(DATA_RAW / "train.csv")[["msno", TARGET]]
    expiry_feb   = build_expiry_features(tx, CUTOFF_FEB)
    churn_lag_feb = pd.DataFrame({"msno": labels_feb["msno"], "prev_churn": 0})  # sin etiquetas previas

    df_feb = build_period_features(labels_feb, tx_feats, log_feb, expiry_feb, churn_lag_feb, CUTOFF_FEB)
    X_feb  = df_feb[FEATURE_COLS].values.astype(np.float32)
    y_feb  = df_feb[TARGET].values
    print(f"  {len(df_feb):,} filas | churn: {y_feb.mean():.2%}")

    # ── PERÍODO MARZO ─────────────────────────────────────────────────────────
    print("\n── Período MARZO (train_v2.csv) ──")
    labels_mar    = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", TARGET]]
    expiry_mar    = build_expiry_features(tx, CUTOFF_MAR)
    churn_lag_mar = build_churn_lag(CUTOFF_MAR)  # etiquetas de febrero disponibles

    df_mar = build_period_features(labels_mar, tx_feats, log_mar, expiry_mar, churn_lag_mar, CUTOFF_MAR)
    X_mar  = df_mar[FEATURE_COLS].values.astype(np.float32)
    y_mar  = df_mar[TARGET].values
    print(f"  {len(df_mar):,} filas | churn: {y_mar.mean():.2%}")

    # ── VALIDACIÓN TEMPORAL HONESTA ───────────────────────────────────────────
    print("\n── Validación temporal: entrenar feb → evaluar mar ──")
    model_feb = train_lgbm(X_feb, y_feb, best_params)
    val_proba = model_feb.predict_proba(X_mar)[:, 1]
    print_metrics("Entrenado feb / Validado mar (métrica honesta):", y_mar, val_proba)

    # ── MODELO FINAL: feb + mar combinados ────────────────────────────────────
    print("\n── Modelo final: feb + mar combinados ──")
    X_all = np.vstack([X_feb, X_mar])
    y_all = np.concatenate([y_feb, y_mar])
    print(f"  {len(y_all):,} filas totales | churn rate: {y_all.mean():.2%}")

    model_final = train_lgbm(X_all, y_all, best_params)
    joblib.dump(
        {"model": model_final, "features": FEATURE_COLS},
        MODELS_DIR / "lgbm_temporal.joblib",
    )
    print(f"  Guardado en models/lgbm_temporal.joblib")

    # ── SUBMISSION: abril ─────────────────────────────────────────────────────
    print("\n── Submission (abr 2017) ──")
    sub           = pd.read_csv(DATA_RAW / "sample_submission_v2.csv")[["msno"]]
    expiry_sub    = build_expiry_features(tx, CUTOFF_MAR)   # misma ref que mar training
    churn_lag_sub = build_churn_lag(CUTOFF_APR)              # etiquetas de marzo disponibles

    df_sub = build_period_features(sub, tx_feats, log_mar, expiry_sub, churn_lag_sub, CUTOFF_MAR)
    print(f"  {len(df_sub):,} filas | nulos: {df_sub[FEATURE_COLS].isnull().sum().sum()}")

    proba = model_final.predict_proba(df_sub[FEATURE_COLS].values.astype(np.float32))[:, 1]
    print(f"  mean: {proba.mean():.4f} | min: {proba.min():.4f} | max: {proba.max():.4f}")

    out_path = SUBMISSIONS_DIR / "submission_v6_temporal.csv"
    pd.DataFrame({"msno": df_sub["msno"], "is_churn": proba}).to_csv(out_path, index=False)
    print(f"\n✓ Submission lista: {len(df_sub):,} filas → {out_path}")


if __name__ == "__main__":
    main()
