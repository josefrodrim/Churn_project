"""
Genera el archivo de submission para Kaggle.
Usuarios: sample_submission_v2.csv (907,471 — predicción abril 2017)
Modelo  : models/lgbm_tuned.joblib
Salida  : submissions/submission_lgbm_tuned.csv
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

PREDICTION_DATE_SUB = pd.Timestamp("2017-04-30")

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import load_log_features, build_log_features
from src.features.build_features_05 import impute_missing
from src.models.train_06 import FEATURE_COLS


def build_member_features_sub(filepath: Path = DATA_RAW / "members_v3.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["gender"] = df["gender"].fillna("unknown")
    df["gender_enc"] = df["gender"].map({"male": 0, "female": 1, "unknown": 2})
    df["bd_valid"] = df["bd"].between(1, 99).astype(int)
    df["age"] = np.where(df["bd_valid"], df["bd"], np.nan)
    df["tenure_days"] = (PREDICTION_DATE_SUB - df["registration_init_time"]).dt.days
    return df[["msno", "city", "registered_via", "gender_enc", "age", "bd_valid", "tenure_days"]]


def build_submission_features() -> pd.DataFrame:
    print("── Cargando usuarios del submission ──")
    sub = pd.read_csv(DATA_RAW / "sample_submission_v2.csv")[["msno"]]
    print(f"  {len(sub):,} usuarios")

    print("\n── Features de transacciones (tx + tx_v2) ──")
    tx = load_transactions()
    tx_feats = build_user_features(tx)
    print(f"  {len(tx_feats):,} usuarios, {tx_feats.shape[1]-1} features")

    print("\n── Features de members (fecha ref: 2017-04-30) ──")
    member_feats = build_member_features_sub()
    print(f"  {len(member_feats):,} usuarios, {member_feats.shape[1]-1} features")

    print("\n── Features de user_logs (caché) ──")
    raw_agg = load_log_features()
    log_feats = build_log_features(raw_agg)
    print(f"  {len(log_feats):,} usuarios, {log_feats.shape[1]-1} features")

    print("\n── Uniendo tablas ──")
    df = sub.copy()
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

    nulls = df[FEATURE_COLS].isnull().sum()
    print(f"  Shape: {df.shape} | Nulos en features: {nulls[nulls > 0].to_dict()}")

    df = impute_missing(df)
    print(f"  Nulos tras imputación: {df[FEATURE_COLS].isnull().sum().sum()}")
    return df


def main():
    print("── Cargando modelo tuneado ──")
    artifact = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")
    model    = artifact["model"]
    features = artifact["features"]
    print(f"  Features del modelo: {len(features)}")

    df = build_submission_features()

    print("\n── Prediciendo probabilidades ──")
    X = df[features].values
    proba = model.predict_proba(X)[:, 1]
    print(f"  Distribución — min: {proba.min():.4f} | mean: {proba.mean():.4f} | max: {proba.max():.4f}")

    print("\n── Guardando submission ──")
    out_path = SUBMISSIONS_DIR / "submission_lgbm_tuned.csv"
    pd.DataFrame({"msno": df["msno"], "is_churn": proba}).to_csv(out_path, index=False)
    print(f"\n✓ Submission lista: {len(df):,} filas → {out_path}")
    return out_path


if __name__ == "__main__":
    main()
