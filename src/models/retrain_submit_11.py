"""
Submission v2: entrena en train + train_v2 con user_logs hasta marzo 2017.
Mejoras sobre predict_10:
  1. user_logs_agg_mar.parquet — incluye user_logs_v2 (hasta 2017-03-31)
  2. Modelo entrenado en ~1.96M filas (train feb + train_v2 mar) en lugar de ~793K
Salida: submissions/submission_combined_v2.csv
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

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROC      = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

PREDICTION_DATE_MAR = pd.Timestamp("2017-03-31")
PREDICTION_DATE_APR = pd.Timestamp("2017-04-30")

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import (
    load_log_features, build_log_features,
    MAX_DATE_MAR, RECENT_CUTOFF_MAR,
)
from src.features.build_features_05 import impute_missing
from src.models.train_06 import FEATURE_COLS, TARGET, RANDOM_STATE


# ── HELPERS ───────────────────────────────────────────────────────────────────

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


def join_features(labels_df, tx_feats, member_feats, log_feats_raw, prediction_date) -> pd.DataFrame:
    log_feats = build_log_features(log_feats_raw, prediction_date=prediction_date)
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
    return impute_missing(df)


# ── DATOS ─────────────────────────────────────────────────────────────────────

def load_combined_train(tx_feats, raw_agg_mar) -> tuple[pd.DataFrame, pd.Series]:
    print("── Train set (feb 2017) — cargando features existentes ──")
    df_feb = pd.read_parquet(DATA_PROC / "features_train.parquet")
    print(f"  {len(df_feb):,} filas | churn rate: {df_feb[TARGET].mean():.2%}")

    print("\n── Train_v2 (mar 2017) — construyendo features con logs hasta marzo ──")
    labels_mar = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", TARGET]]
    member_feats_mar = build_member_features(PREDICTION_DATE_MAR)
    df_mar = join_features(labels_mar, tx_feats, member_feats_mar, raw_agg_mar, PREDICTION_DATE_MAR)
    print(f"  {len(df_mar):,} filas | churn rate: {df_mar[TARGET].mean():.2%}")

    df_all = pd.concat([df_feb, df_mar], ignore_index=True)
    X = df_all[FEATURE_COLS].values.astype(np.float32)
    y = df_all[TARGET].values
    print(f"\n  Combinado: {len(df_all):,} filas | churn rate: {y.mean():.2%}")
    return X, y


def load_submission_features(tx_feats, raw_agg_mar) -> pd.DataFrame:
    print("\n── Submission features (apr 2017) ──")
    sub = pd.read_csv(DATA_RAW / "sample_submission_v2.csv")[["msno"]]
    member_feats_apr = build_member_features(PREDICTION_DATE_APR)
    df = join_features(sub, tx_feats, member_feats_apr, raw_agg_mar, PREDICTION_DATE_APR)
    print(f"  {len(df):,} filas | nulos: {df[FEATURE_COLS].isnull().sum().sum()}")
    return df


# ── TRAIN ─────────────────────────────────────────────────────────────────────

def train_lgbm(X_train, y_train, best_params: dict) -> LGBMClassifier:
    model = LGBMClassifier(
        **best_params,
        is_unbalance=True,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("RETRAIN + SUBMIT v2")
    print("═" * 60)

    print("\n── Cargando user_logs hasta marzo (cache) ──")
    raw_agg_mar = load_log_features(max_date=MAX_DATE_MAR, recent_cutoff=RECENT_CUTOFF_MAR)
    print(f"  {len(raw_agg_mar):,} usuarios")

    print("\n── Cargando transacciones ──")
    tx = load_transactions()
    tx_feats = build_user_features(tx)
    print(f"  {len(tx_feats):,} usuarios")

    X, y = load_combined_train(tx_feats, raw_agg_mar)

    print("\n── Cargando mejores hiperparámetros (lgbm_tuned) ──")
    artifact = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")
    best_params = artifact["best_params"]
    for k, v in best_params.items():
        print(f"  {k:25s}: {v}")

    print("\n── Entrenando modelo combinado ──")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE)
    model = train_lgbm(X_tr, y_tr, best_params)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    val_pr  = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"  Validación (10%) — ROC-AUC: {val_auc:.5f} | PR-AUC: {val_pr:.5f}")

    model_path = MODELS_DIR / "lgbm_combined.joblib"
    joblib.dump({"model": model, "features": FEATURE_COLS, "val_auc": val_auc}, model_path)
    print(f"  Guardado en {model_path.name}")

    df_sub = load_submission_features(tx_feats, raw_agg_mar)

    print("\n── Prediciendo ──")
    proba = model.predict_proba(df_sub[FEATURE_COLS].values.astype(np.float32))[:, 1]
    print(f"  min: {proba.min():.4f} | mean: {proba.mean():.4f} | max: {proba.max():.4f}")

    out_path = SUBMISSIONS_DIR / "submission_combined_v2.csv"
    pd.DataFrame({"msno": df_sub["msno"], "is_churn": proba}).to_csv(out_path, index=False)
    print(f"\n✓ Submission lista: {len(df_sub):,} filas → {out_path}")


if __name__ == "__main__":
    main()
