"""
Submission v4 — expiry features + churn lag.

Nuevas features sobre las existentes de v3:
  - days_until_expire   : días entre última fecha de expiración y prediction_date
  - is_expired          : la membresía ya expiró antes de prediction_date
  - auto_renew_at_expire: tenía auto-renew en la transacción de mayor expiración
  - cancel_at_expire    : era un cancel en esa misma transacción
  - n_renewals          : cuántas veces renovó (non-cancel transactions)
  - prev_churn          : churn del mes anterior (feature de lag)

Entrenamiento: train_v2 (mar 2017), logs hasta mar 31, days_since_last con
referencia fija LOG_CUTOFF_DATE (fix de v3).
Salida: submissions/submission_v4_expiry.csv
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

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

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import (
    load_log_features, build_log_features,
    MAX_DATE_MAR, RECENT_CUTOFF_MAR,
)
from src.features.build_features_05 import impute_missing
from src.models.train_06 import FEATURE_COLS, TARGET, RANDOM_STATE

# Features base + 6 nuevas
FEATURE_COLS_V4 = FEATURE_COLS + [
    "days_until_expire",
    "is_expired",
    "auto_renew_at_expire",
    "cancel_at_expire",
    "n_renewals",
    "prev_churn",
]


# ── NUEVAS FEATURES ───────────────────────────────────────────────────────────

def build_expiry_features(tx: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.DataFrame:
    """Features basadas en membership_expire_date — la señal más directa de churn."""
    df = tx.copy()
    # membership_expire_date already comes as datetime from load_transactions()
    df["expire_date"] = pd.to_datetime(df["membership_expire_date"], errors="coerce")

    # Transacción con la fecha de expiración más lejana por usuario
    df_valid = df.dropna(subset=["expire_date"])
    idx_max_expire = df_valid.groupby("msno")["expire_date"].idxmax()
    last_expire_tx = df_valid.loc[idx_max_expire, ["msno", "expire_date", "is_auto_renew", "is_cancel"]].copy()
    last_expire_tx.columns = ["msno", "last_expire_date", "auto_renew_at_expire", "cancel_at_expire"]

    last_expire_tx["days_until_expire"] = (
        last_expire_tx["last_expire_date"] - prediction_date
    ).dt.days
    last_expire_tx["is_expired"] = (last_expire_tx["days_until_expire"] < 0).astype(int)

    # Número de renovaciones (transacciones que no son cancel)
    n_renewals = (
        df[df["is_cancel"] == 0].groupby("msno").size().reset_index(name="n_renewals")
    )

    result = last_expire_tx.merge(n_renewals, on="msno", how="left")
    result["n_renewals"] = result["n_renewals"].fillna(0)

    return result[[
        "msno", "days_until_expire", "is_expired",
        "auto_renew_at_expire", "cancel_at_expire", "n_renewals",
    ]]


def build_churn_lag(prediction_date: pd.Timestamp) -> pd.DataFrame:
    """Churn del mes anterior como feature. Cero para usuarios nuevos."""
    if prediction_date >= pd.Timestamp("2017-04-01"):
        # Predecimos abril → usamos etiquetas de marzo (train_v2)
        src = DATA_RAW / "train_v2.csv"
    else:
        # Predecimos marzo → usamos etiquetas de febrero (train)
        src = DATA_RAW / "train.csv"
    labels = pd.read_csv(src)[["msno", "is_churn"]]
    labels.columns = ["msno", "prev_churn"]
    return labels


def build_member_features(prediction_date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "members_v3.csv")
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["gender"] = df["gender"].fillna("unknown")
    df["gender_enc"] = df["gender"].map({"male": 0, "female": 1, "unknown": 2})
    df["bd_valid"] = df["bd"].between(1, 99).astype(int)
    df["age"] = np.where(df["bd_valid"], df["bd"], np.nan)
    df["tenure_days"] = (prediction_date - df["registration_init_time"]).dt.days
    return df[["msno", "city", "registered_via", "gender_enc", "age", "bd_valid", "tenure_days"]]


def join_all_features(
    labels_df, tx, tx_feats, member_feats, raw_agg, expiry_feats, churn_lag, prediction_date
) -> pd.DataFrame:
    log_feats = build_log_features(
        raw_agg,
        prediction_date=LOG_CUTOFF_DATE,
        days_since_ref=LOG_CUTOFF_DATE,
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

    # Nuevas features
    df = df.merge(expiry_feats, on="msno", how="left")
    df = df.merge(churn_lag, on="msno", how="left")

    # Imputar nuevas features
    df["days_until_expire"]   = df["days_until_expire"].fillna(df["days_until_expire"].median())
    df["is_expired"]          = df["is_expired"].fillna(1).astype(int)
    df["auto_renew_at_expire"]= df["auto_renew_at_expire"].fillna(0).astype(int)
    df["cancel_at_expire"]    = df["cancel_at_expire"].fillna(0).astype(int)
    df["n_renewals"]          = df["n_renewals"].fillna(0)
    df["prev_churn"]          = df["prev_churn"].fillna(0).astype(int)

    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("SUBMIT v4 — expiry features + churn lag")
    print("═" * 60)

    print("\n── Cargando datos base ──")
    raw_agg_mar = load_log_features(max_date=MAX_DATE_MAR, recent_cutoff=RECENT_CUTOFF_MAR)
    print(f"  user_logs cache: {len(raw_agg_mar):,} usuarios")

    tx = load_transactions()
    tx_feats = build_user_features(tx)
    print(f"  transacciones: {len(tx_feats):,} usuarios")

    print("\n── Construyendo nuevas features ──")
    # LOG_CUTOFF_DATE como referencia fija para training Y submission — mismo offset = 0
    expiry_train = build_expiry_features(tx, LOG_CUTOFF_DATE)
    churn_lag_train = build_churn_lag(PREDICTION_DATE_MAR)
    print(f"  expiry features: {len(expiry_train):,} usuarios")
    print(f"  churn lag: {len(churn_lag_train):,} usuarios con etiqueta previa")
    print(f"  prev_churn rate: {churn_lag_train['prev_churn'].mean():.2%}")

    # ── TRAIN ────────────────────────────────────────────────────────────────
    print("\n── Train set: train_v2 (mar 2017) ──")
    labels_mar   = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", TARGET]]
    member_feats = build_member_features(PREDICTION_DATE_MAR)

    df_train = join_all_features(
        labels_mar, tx, tx_feats, member_feats, raw_agg_mar,
        expiry_train, churn_lag_train, PREDICTION_DATE_MAR,
    )
    X = df_train[FEATURE_COLS_V4].values.astype(np.float32)
    y = df_train[TARGET].values
    print(f"  {len(df_train):,} filas | churn: {y.mean():.2%} | features: {len(FEATURE_COLS_V4)}")

    # ── MODELO ───────────────────────────────────────────────────────────────
    best_params = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")["best_params"]

    print("\n── Entrenando modelo v4 ──")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
    )
    model = LGBMClassifier(**best_params, is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1)
    model.fit(X_tr, y_tr)

    val_proba = model.predict_proba(X_val)[:, 1]
    print(f"  Val ROC-AUC : {roc_auc_score(y_val, val_proba):.5f}")
    print(f"  Val PR-AUC  : {average_precision_score(y_val, val_proba):.5f}")

    # Top features por importancia
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS_V4)
    print("\n  Top 10 features:")
    for feat, imp in importances.nlargest(10).items():
        print(f"    {feat:30s}: {imp:,}")

    joblib.dump({"model": model, "features": FEATURE_COLS_V4}, MODELS_DIR / "lgbm_v4.joblib")

    # ── SUBMISSION ────────────────────────────────────────────────────────────
    print("\n── Submission features (apr 2017) ──")
    sub = pd.read_csv(DATA_RAW / "sample_submission_v2.csv")[["msno"]]
    expiry_sub    = build_expiry_features(tx, LOG_CUTOFF_DATE)  # misma ref que training
    churn_lag_sub = build_churn_lag(PREDICTION_DATE_APR)
    member_sub    = build_member_features(PREDICTION_DATE_APR)

    df_sub = join_all_features(
        sub, tx, tx_feats, member_sub, raw_agg_mar,
        expiry_sub, churn_lag_sub, PREDICTION_DATE_APR,
    )
    print(f"  {len(df_sub):,} filas | nulos: {df_sub[FEATURE_COLS_V4].isnull().sum().sum()}")

    print("\n── Prediciendo ──")
    proba = model.predict_proba(df_sub[FEATURE_COLS_V4].values.astype(np.float32))[:, 1]
    print(f"  min: {proba.min():.4f} | mean: {proba.mean():.4f} | max: {proba.max():.4f}")

    out_path = SUBMISSIONS_DIR / "submission_v4_expiry.csv"
    pd.DataFrame({"msno": df_sub["msno"], "is_churn": proba}).to_csv(out_path, index=False)
    print(f"\n✓ Submission lista: {len(df_sub):,} filas → {out_path}")


if __name__ == "__main__":
    main()
