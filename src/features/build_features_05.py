"""
Feature engineering: construye la matriz de features por usuario uniendo
transactions, members y user_logs. Salida: data/processed/features_train.parquet

Ejecución:
    python src/features/build_features_05.py

Re-runs reutilizan el caché de user_logs (data/processed/user_logs_agg.parquet)
para evitar reprocesar los 28 GB. Borra el archivo para forzar recálculo.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

PREDICTION_DATE = pd.Timestamp("2017-02-28")

FEATURES_OUT = DATA_PROC / "features_train.parquet"

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import load_log_features, build_log_features


# ── MEMBERS ───────────────────────────────────────────────────────────────────

def build_member_features(filepath: Path = DATA_RAW / "members_v3.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["gender"] = df["gender"].fillna("unknown")
    df["gender_enc"] = df["gender"].map({"male": 0, "female": 1, "unknown": 2})
    df["bd_valid"] = df["bd"].between(1, 99).astype(int)
    df["age"] = np.where(df["bd_valid"], df["bd"], np.nan)
    df["tenure_days"] = (PREDICTION_DATE - df["registration_init_time"]).dt.days

    keep = ["msno", "city", "registered_via", "gender_enc", "age", "bd_valid", "tenure_days"]
    return df[keep]


# ── TRANSACTIONS ──────────────────────────────────────────────────────────────

def get_transaction_features() -> pd.DataFrame:
    print("Cargando transacciones...")
    tx = load_transactions()
    print(f"  {len(tx):,} filas | construyendo features por usuario...")
    return build_user_features(tx)


# ── USER LOGS ─────────────────────────────────────────────────────────────────

def get_log_features() -> pd.DataFrame:
    raw_agg = load_log_features()   # caché manejado en user_logs_04.load_log_features
    return build_log_features(raw_agg)


# ── JOIN Y LIMPIEZA ───────────────────────────────────────────────────────────

def join_all(
    labels: pd.DataFrame,
    tx_feats: pd.DataFrame,
    member_feats: pd.DataFrame,
    log_feats: pd.DataFrame,
) -> pd.DataFrame:
    df = labels.copy()                                    # base: train users

    df = df.merge(tx_feats, on="msno", how="left")        # 100% cobertura
    df = df.merge(member_feats, on="msno", how="left")    # ~88% cobertura
    df = df.merge(
        log_feats[[
            "msno", "n_days", "avg_daily_secs", "avg_daily_completed",
            "avg_daily_unq", "completion_ratio", "days_since_last", "listening_trend",
        ]],
        on="msno", how="left",                            # ~88% cobertura
    )

    # Flags de cobertura
    df["has_member_record"] = df["city"].notna().astype(int)
    df["has_log_record"] = df["n_days"].notna().astype(int)

    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Numéricas de members: mediana de quienes sí tienen registro
    for col in ["city", "registered_via", "gender_enc", "age", "bd_valid", "tenure_days"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Numéricas de logs
    count_cols = ["n_days"]
    rate_cols = [
        "avg_daily_secs", "avg_daily_completed", "avg_daily_unq",
        "completion_ratio", "days_since_last", "listening_trend",
    ]
    for col in count_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in rate_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Numéricas de transacciones (no deberían faltar, pero por si acaso)
    tx_cols = [c for c in df.columns if c not in
               ["msno", "is_churn", "has_member_record", "has_log_record"]]
    for col in tx_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


# ── MAIN ──────────────────────────────────────────────────────────────────────

def build_and_save() -> pd.DataFrame:
    print("── Cargando etiquetas de train ──")
    labels = pd.read_csv(DATA_RAW / "train.csv")[["msno", "is_churn"]]
    print(f"  {len(labels):,} usuarios | churn rate: {labels['is_churn'].mean():.2%}")

    print("\n── Features de transacciones ──")
    tx_feats = get_transaction_features()
    print(f"  {len(tx_feats):,} usuarios, {tx_feats.shape[1]-1} features")

    print("\n── Features de members ──")
    member_feats = build_member_features()
    print(f"  {len(member_feats):,} usuarios, {member_feats.shape[1]-1} features")

    print("\n── Features de user_logs ──")
    log_feats = get_log_features()
    print(f"  {len(log_feats):,} usuarios, {log_feats.shape[1]-1} features")

    print("\n── Uniendo tablas ──")
    df = join_all(labels, tx_feats, member_feats, log_feats)
    print(f"  Shape antes de imputar: {df.shape}")
    print(f"  Nulos por columna (top 10):")
    nulls = df.isnull().sum().sort_values(ascending=False)
    print(nulls[nulls > 0].head(10).to_string())

    print("\n── Imputando valores faltantes ──")
    df = impute_missing(df)
    remaining_nulls = df.isnull().sum().sum()
    print(f"  Nulos restantes: {remaining_nulls}")

    print(f"\n── Guardando en {FEATURES_OUT} ──")
    df.to_parquet(FEATURES_OUT, index=False)

    print(f"\n✓ Feature matrix lista: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"  Features: {[c for c in df.columns if c not in ['msno', 'is_churn']]}")
    return df


if __name__ == "__main__":
    build_and_save()
