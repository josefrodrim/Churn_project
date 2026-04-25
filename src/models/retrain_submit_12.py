"""
Submission v3 — fix del offset en days_since_last.

Problema en v2: mezclar train(feb) + train_v2(mar) genera dos escalas distintas
de days_since_last. Además, submission usa Apr30 como referencia pero los logs
solo llegan a Mar31, introduciendo un offset de ≥30 días no visto en training.

Fix:
  1. Entrenar solo en train_v2 (mar 2017) — mismo período que los logs.
  2. Usar LOG_CUTOFF (2017-03-31) como referencia para days_since_last en
     training Y submission — offset = 0 en ambos.
  3. Logs hasta Mar31 (user_logs_agg_mar.parquet) — más recientes.

Salida: submissions/submission_v3_fixed.csv
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
LOG_CUTOFF_DATE     = pd.Timestamp("2017-03-31")  # mismo para train y submission

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


def join_features(labels_df, tx_feats, member_feats, raw_agg) -> pd.DataFrame:
    # days_since_ref = LOG_CUTOFF_DATE siempre — mismo punto de referencia en training y submission
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
    return impute_missing(df)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("SUBMIT v3 — fix days_since_last offset")
    print("═" * 60)

    print("\n── Cargando user_logs hasta marzo (cache) ──")
    raw_agg_mar = load_log_features(max_date=MAX_DATE_MAR, recent_cutoff=RECENT_CUTOFF_MAR)
    print(f"  {len(raw_agg_mar):,} usuarios")

    print("\n── Cargando transacciones ──")
    tx = load_transactions()
    tx_feats = build_user_features(tx)
    print(f"  {len(tx_feats):,} usuarios")

    # ── TRAIN: solo train_v2 (mar 2017) ──────────────────────────────────────
    print("\n── Train set: train_v2 (mar 2017) ──")
    labels_mar = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", TARGET]]
    member_feats_mar = build_member_features(PREDICTION_DATE_MAR)
    df_train = join_features(labels_mar, tx_feats, member_feats_mar, raw_agg_mar)
    X = df_train[FEATURE_COLS].values.astype(np.float32)
    y = df_train[TARGET].values
    print(f"  {len(df_train):,} filas | churn rate: {y.mean():.2%}")

    # ── MODELO ───────────────────────────────────────────────────────────────
    print("\n── Cargando mejores hiperparámetros (lgbm_tuned) ──")
    artifact    = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")
    best_params = artifact["best_params"]

    print("\n── Entrenando modelo v3 ──")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=RANDOM_STATE
    )
    model = LGBMClassifier(**best_params, is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1)
    model.fit(X_tr, y_tr)

    val_proba = model.predict_proba(X_val)[:, 1]
    print(f"  Val ROC-AUC : {roc_auc_score(y_val, val_proba):.5f}")
    print(f"  Val PR-AUC  : {average_precision_score(y_val, val_proba):.5f}")

    model_path = MODELS_DIR / "lgbm_v3.joblib"
    joblib.dump({"model": model, "features": FEATURE_COLS}, model_path)
    print(f"  Guardado en {model_path.name}")

    # ── SUBMISSION FEATURES ──────────────────────────────────────────────────
    print("\n── Submission features (ref: 2017-03-31) ──")
    sub = pd.read_csv(DATA_RAW / "sample_submission_v2.csv")[["msno"]]
    member_feats_sub = build_member_features(PREDICTION_DATE_APR)
    df_sub = join_features(sub, tx_feats, member_feats_sub, raw_agg_mar)
    print(f"  {len(df_sub):,} filas | nulos: {df_sub[FEATURE_COLS].isnull().sum().sum()}")

    # ── PREDICCIÓN ───────────────────────────────────────────────────────────
    print("\n── Prediciendo ──")
    proba = model.predict_proba(df_sub[FEATURE_COLS].values.astype(np.float32))[:, 1]
    print(f"  min: {proba.min():.4f} | mean: {proba.mean():.4f} | max: {proba.max():.4f}")

    out_path = SUBMISSIONS_DIR / "submission_v3_fixed.csv"
    pd.DataFrame({"msno": df_sub["msno"], "is_churn": proba}).to_csv(out_path, index=False)
    print(f"\n✓ Submission lista: {len(df_sub):,} filas → {out_path}")


if __name__ == "__main__":
    main()
