"""
Submission v5 — kitchen sink: multiwindow logs + tx recency + calibración.

Sobre v4 añade:
  - user_logs ventanas 7d y 90d (cache user_logs_agg_mar_v2.parquet ~15 min rebuild)
  - n_days_7d, secs_7d, n_days_90d, secs_90d, trend_7d, trend_7d_vs_30d
  - days_since_last_tx, had_tx_last_7d, had_tx_last_30d, n_tx_last_30d
  - cancel_before_expire (cancel + membresía aún activa = churn diferido seguro)
  - Calibración isotónica (ajusta probabilidades en holdout)

Total: 36 + 12 = 48 features + calibración.
Salida: submissions/submission_v5_full.csv
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_RAW        = ROOT / "data" / "raw"
DATA_PROC       = ROOT / "data" / "processed"
MODELS_DIR      = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

PREDICTION_DATE_MAR = pd.Timestamp("2017-03-31")
PREDICTION_DATE_APR = pd.Timestamp("2017-04-30")
LOG_CUTOFF_DATE     = pd.Timestamp("2017-03-31")
LOG_CUTOFF_INT      = 20170331

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import (
    load_log_features, build_log_features,
    MAX_DATE_MAR, RECENT_CUTOFF_MAR,
    EXTRA_CUTOFFS_MAR_V2,
)
from src.features.build_features_05 import impute_missing
from src.models.train_06 import FEATURE_COLS, TARGET, RANDOM_STATE
from src.models.retrain_submit_13 import (
    build_member_features, build_expiry_features, build_churn_lag,
)

# ── NUEVAS FEATURES V5 ────────────────────────────────────────────────────────

EXTRA_LOG_FEATURES = [
    "n_days_7d", "secs_per_day_7d",
    "n_days_90d", "secs_per_day_90d",
    "trend_7d", "trend_7d_vs_30d",
]

TX_RECENCY_FEATURES = [
    "days_since_last_tx", "had_tx_last_7d",
    "had_tx_last_30d", "n_tx_last_30d",
]

FEATURE_COLS_V5 = (
    FEATURE_COLS
    + ["days_until_expire", "is_expired", "auto_renew_at_expire",
       "cancel_at_expire", "n_renewals", "prev_churn"]
    + TX_RECENCY_FEATURES
    + ["cancel_before_expire"]
    + EXTRA_LOG_FEATURES
)


def build_multiwindow_log_features(raw_agg_v2: pd.DataFrame) -> pd.DataFrame:
    """Deriva features de las ventanas 7d y 90d del cache v2."""
    df = raw_agg_v2.copy()
    eps = 1  # evitar división por cero

    df["n_days_7d"]  = df.get("r7d_n_days", 0).fillna(0)
    df["n_days_90d"] = df.get("r90d_n_days", 0).fillna(0)

    df["secs_per_day_7d"]  = df.get("r7d_secs_sum", 0).fillna(0) / (df["n_days_7d"] + eps)
    df["secs_per_day_90d"] = df.get("r90d_secs_sum", 0).fillna(0) / (df["n_days_90d"] + eps)

    overall_daily = df["total_secs_sum"] / (df["n_days"] + eps)
    secs_30d_per_day = df["recent_total_secs_sum"] / (df["recent_n_days"] + eps)
    df["trend_7d"]       = df["secs_per_day_7d"] - overall_daily
    df["trend_7d_vs_30d"] = df["secs_per_day_7d"] - secs_30d_per_day

    return df[["msno"] + EXTRA_LOG_FEATURES]


def build_tx_recency_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Recencia de transacciones relativa al LOG_CUTOFF_DATE."""
    cutoff_30d = LOG_CUTOFF_DATE - pd.Timedelta(days=30)
    cutoff_7d  = LOG_CUTOFF_DATE - pd.Timedelta(days=7)

    tx_dt = pd.to_datetime(tx["transaction_date"], errors="coerce")

    last_tx = tx_dt.groupby(tx["msno"]).max().reset_index()
    last_tx.columns = ["msno", "last_tx_date"]
    last_tx["days_since_last_tx"] = (LOG_CUTOFF_DATE - last_tx["last_tx_date"]).dt.days.clip(lower=0)

    mask30 = tx_dt >= cutoff_30d
    mask7  = tx_dt >= cutoff_7d

    n30 = tx[mask30].groupby("msno").size().reset_index(name="n_tx_last_30d")
    n7  = tx[mask7].groupby("msno").size().reset_index(name="n_tx_last_7d")

    result = last_tx.merge(n30, on="msno", how="left").merge(n7, on="msno", how="left")
    result["n_tx_last_30d"] = result["n_tx_last_30d"].fillna(0)
    result["n_tx_last_7d"]  = result["n_tx_last_7d"].fillna(0)
    result["had_tx_last_30d"] = (result["n_tx_last_30d"] > 0).astype(int)
    result["had_tx_last_7d"]  = (result["n_tx_last_7d"] > 0).astype(int)

    return result[["msno"] + TX_RECENCY_FEATURES]


def join_all_v5(labels_df, tx_feats, member_feats, raw_agg_v2,
                expiry_feats, churn_lag, tx_recency) -> pd.DataFrame:
    log_base = build_log_features(
        raw_agg_v2,
        prediction_date=LOG_CUTOFF_DATE,
        days_since_ref=LOG_CUTOFF_DATE,
    )
    log_extra = build_multiwindow_log_features(raw_agg_v2)

    df = labels_df.copy()
    df = df.merge(tx_feats, on="msno", how="left")
    df = df.merge(member_feats, on="msno", how="left")
    df = df.merge(
        log_base[["msno", "n_days", "avg_daily_secs", "avg_daily_completed",
                  "avg_daily_unq", "completion_ratio", "days_since_last", "listening_trend"]],
        on="msno", how="left",
    )
    df["has_member_record"] = df["city"].notna().astype(int)
    df["has_log_record"]    = df["n_days"].notna().astype(int)
    df = impute_missing(df)

    df = df.merge(expiry_feats, on="msno", how="left")
    df = df.merge(churn_lag,    on="msno", how="left")
    df = df.merge(tx_recency,   on="msno", how="left")
    df = df.merge(log_extra,    on="msno", how="left")

    # Imputar nuevas features
    df["days_until_expire"]    = df["days_until_expire"].fillna(df["days_until_expire"].median())
    df["is_expired"]           = df["is_expired"].fillna(1).astype(int)
    df["auto_renew_at_expire"] = df["auto_renew_at_expire"].fillna(0).astype(int)
    df["cancel_at_expire"]     = df["cancel_at_expire"].fillna(0).astype(int)
    df["n_renewals"]           = df["n_renewals"].fillna(0)
    df["prev_churn"]           = df["prev_churn"].fillna(0).astype(int)
    df["days_since_last_tx"]   = df["days_since_last_tx"].fillna(df["days_since_last_tx"].median())
    df["had_tx_last_7d"]       = df["had_tx_last_7d"].fillna(0).astype(int)
    df["had_tx_last_30d"]      = df["had_tx_last_30d"].fillna(0).astype(int)
    df["n_tx_last_30d"]        = df["n_tx_last_30d"].fillna(0)
    for col in EXTRA_LOG_FEATURES:
        df[col] = df[col].fillna(0)

    # cancel_before_expire: canceló pero membresía aún vigente = churn diferido
    df["cancel_before_expire"] = (
        (df["last_is_cancel"] == 1) & (df["days_until_expire"] > 0)
    ).astype(int)

    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("SUBMIT v5 — multiwindow + tx recency + calibración")
    print("═" * 60)

    print("\n── Cargando user_logs multiventana (7d + 30d + 90d) ──")
    raw_agg_v2 = load_log_features(
        max_date=MAX_DATE_MAR,
        recent_cutoff=RECENT_CUTOFF_MAR,
        extra_cutoffs=EXTRA_CUTOFFS_MAR_V2,
    )
    print(f"  {len(raw_agg_v2):,} usuarios | cols: {raw_agg_v2.shape[1]}")

    print("\n── Cargando transacciones ──")
    tx = load_transactions()
    tx_feats   = build_user_features(tx)
    tx_recency = build_tx_recency_features(tx)
    print(f"  {len(tx_feats):,} usuarios tx | {len(tx_recency):,} usuarios recency")

    print("\n── Construyendo features de entrenamiento (train_v2, mar 2017) ──")
    labels_mar   = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", TARGET]]
    member_mar   = build_member_features(PREDICTION_DATE_MAR)
    expiry_train = build_expiry_features(tx, LOG_CUTOFF_DATE)
    churn_lag_tr = build_churn_lag(PREDICTION_DATE_MAR)

    df_train = join_all_v5(
        labels_mar, tx_feats, member_mar, raw_agg_v2,
        expiry_train, churn_lag_tr, tx_recency,
    )
    X = df_train[FEATURE_COLS_V5].values.astype(np.float32)
    y = df_train[TARGET].values
    print(f"  {len(df_train):,} filas | churn: {y.mean():.2%} | features: {len(FEATURE_COLS_V5)}")
    print(f"  Nulos: {df_train[FEATURE_COLS_V5].isnull().sum().sum()}")

    print("\n── Entrenando LightGBM v5 ──")
    best_params = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")["best_params"]
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    model = LGBMClassifier(**best_params, is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1)
    model.fit(X_tr, y_tr)

    cal_proba = model.predict_proba(X_cal)[:, 1]
    print(f"  Pre-cal  — ROC-AUC: {roc_auc_score(y_cal, cal_proba):.5f} | "
          f"PR-AUC: {average_precision_score(y_cal, cal_proba):.5f} | "
          f"LogLoss: {log_loss(y_cal, cal_proba):.5f}")

    print("\n── Calibración isotónica ──")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_proba, y_cal)
    cal_proba_cal = calibrator.predict(cal_proba)
    print(f"  Post-cal — ROC-AUC: {roc_auc_score(y_cal, cal_proba_cal):.5f} | "
          f"PR-AUC: {average_precision_score(y_cal, cal_proba_cal):.5f} | "
          f"LogLoss: {log_loss(y_cal, cal_proba_cal):.5f}")

    top10 = pd.Series(model.feature_importances_, index=FEATURE_COLS_V5).nlargest(10)
    print("\n  Top 10 features:")
    for feat, imp in top10.items():
        print(f"    {feat:35s}: {imp:,}")

    joblib.dump(
        {"model": model, "calibrator": calibrator, "features": FEATURE_COLS_V5},
        MODELS_DIR / "lgbm_v5.joblib",
    )

    print("\n── Submission features (apr 2017) ──")
    sub          = pd.read_csv(DATA_RAW / "sample_submission_v2.csv")[["msno"]]
    member_sub   = build_member_features(PREDICTION_DATE_APR)
    expiry_sub   = build_expiry_features(tx, LOG_CUTOFF_DATE)
    churn_lag_sub = build_churn_lag(PREDICTION_DATE_APR)

    df_sub = join_all_v5(
        sub, tx_feats, member_sub, raw_agg_v2,
        expiry_sub, churn_lag_sub, tx_recency,
    )
    print(f"  {len(df_sub):,} filas | nulos: {df_sub[FEATURE_COLS_V5].isnull().sum().sum()}")

    print("\n── Prediciendo + calibrando ──")
    raw_proba   = model.predict_proba(df_sub[FEATURE_COLS_V5].values.astype(np.float32))[:, 1]
    final_proba = calibrator.predict(raw_proba)
    print(f"  Raw   — mean: {raw_proba.mean():.4f}")
    print(f"  Calib — mean: {final_proba.mean():.4f}")

    out_path = SUBMISSIONS_DIR / "submission_v5_full.csv"
    pd.DataFrame({"msno": df_sub["msno"], "is_churn": final_proba}).to_csv(out_path, index=False)
    print(f"\n✓ Submission lista: {len(df_sub):,} filas → {out_path}")


if __name__ == "__main__":
    main()
