"""
Feature engineering para el test set (train_v2.csv — predicción marzo 2017).
Reutiliza la misma lógica de build_features_05.py pero con fecha de predicción
2017-03-31 y etiquetas de train_v2.csv.
Salida: data/processed/features_test.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

PREDICTION_DATE_TEST = pd.Timestamp("2017-03-31")
FEATURES_OUT = DATA_PROC / "features_test.parquet"

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import load_log_features, build_log_features
from src.features.build_features_05 import impute_missing
from src.models.train_06 import FEATURE_COLS


# ── MEMBERS (con fecha de predicción ajustada) ────────────────────────────────

def build_member_features_test(filepath: Path = DATA_RAW / "members_v3.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["gender"] = df["gender"].fillna("unknown")
    df["gender_enc"] = df["gender"].map({"male": 0, "female": 1, "unknown": 2})
    df["bd_valid"] = df["bd"].between(1, 99).astype(int)
    df["age"] = np.where(df["bd_valid"], df["bd"], np.nan)
    df["tenure_days"] = (PREDICTION_DATE_TEST - df["registration_init_time"]).dt.days

    return df[["msno", "city", "registered_via", "gender_enc", "age", "bd_valid", "tenure_days"]]


# ── MAIN ──────────────────────────────────────────────────────────────────────

def build_and_save_test() -> pd.DataFrame:
    print("── Cargando etiquetas de train_v2 (test set) ──")
    labels = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", "is_churn"]]
    print(f"  {len(labels):,} usuarios | churn rate: {labels['is_churn'].mean():.2%}")

    print("\n── Features de transacciones ──")
    tx = load_transactions()
    tx_feats = build_user_features(tx)
    print(f"  {len(tx_feats):,} usuarios, {tx_feats.shape[1]-1} features")

    print("\n── Features de members (fecha ref: 2017-03-31) ──")
    member_feats = build_member_features_test()
    print(f"  {len(member_feats):,} usuarios, {member_feats.shape[1]-1} features")

    print("\n── Features de user_logs (caché) ──")
    raw_agg = load_log_features()
    log_feats = build_log_features(raw_agg)
    print(f"  {len(log_feats):,} usuarios, {log_feats.shape[1]-1} features")

    print("\n── Uniendo tablas ──")
    df = labels.copy()
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

    print(f"  Shape antes de imputar: {df.shape}")
    nulls = df.isnull().sum()
    print(f"  Nulos (top 5): {nulls[nulls > 0].sort_values(ascending=False).head().to_dict()}")

    print("\n── Imputando valores faltantes ──")
    df = impute_missing(df)
    print(f"  Nulos restantes: {df.isnull().sum().sum()}")

    print(f"\n── Guardando en {FEATURES_OUT.name} ──")
    df.to_parquet(FEATURES_OUT, index=False)
    print(f"\n✓ Test features listas: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


if __name__ == "__main__":
    build_and_save_test()
