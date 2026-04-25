"""
EDA — Paso 4: User logs — comportamiento de escucha por usuario.
Procesa user_logs.csv (~392M filas, 28GB) por chunks y agrega a nivel usuario.
Columnas: msno, date, num_25/50/75/985/100, num_unq, total_secs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

PREDICTION_DATE = pd.Timestamp("2017-02-28")
RECENT_CUTOFF = 20170130  # últimos 30 días antes de prediction date (int YYYYMMDD)
CHUNK_SIZE = 2_000_000

SONG_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100"]

sns.set_theme(style="whitegrid")


# ── LOADERS ──────────────────────────────────────────────────────────────────

MAX_DATE = 20170228  # prediction date — exclude any rows after this (avoid leakage)
MAX_SECS = 86400    # sentinel / corrupt values show as ±9.22e15; cap to max 1 day


def _agg_chunk(chunk: pd.DataFrame, max_date: int = MAX_DATE, recent_cutoff: int = RECENT_CUTOFF) -> pd.DataFrame:
    chunk = chunk[chunk["date"] <= max_date].copy()
    if chunk.empty:
        return pd.DataFrame()
    chunk["total_secs"] = chunk["total_secs"].clip(lower=0, upper=MAX_SECS)
    chunk["total_songs"] = chunk[SONG_COLS].sum(axis=1)
    chunk["is_recent"] = (chunk["date"] >= recent_cutoff).astype(int)
    chunk["recent_secs"] = chunk["total_secs"] * chunk["is_recent"]
    return (
        chunk.groupby("msno", sort=False)
        .agg(
            n_days=("date", "count"),
            total_secs_sum=("total_secs", "sum"),
            num_100_sum=("num_100", "sum"),
            num_unq_sum=("num_unq", "sum"),
            total_songs_sum=("total_songs", "sum"),
            max_date=("date", "max"),
            recent_n_days=("is_recent", "sum"),
            recent_total_secs_sum=("recent_secs", "sum"),
        )
        .reset_index()
    )


def aggregate_logs_chunked(filepath: Path, max_date: int = MAX_DATE, recent_cutoff: int = RECENT_CUTOFF) -> pd.DataFrame:
    SUM_COLS = [
        "n_days", "total_secs_sum", "num_100_sum", "num_unq_sum",
        "total_songs_sum", "recent_n_days", "recent_total_secs_sum",
    ]
    parts = []
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=CHUNK_SIZE)):
        part = _agg_chunk(chunk, max_date, recent_cutoff)
        if not part.empty:
            parts.append(part)
        if (i + 1) % 25 == 0:
            print(f"  {filepath.name}: chunk {i+1} ({(i+1)*CHUNK_SIZE/1e6:.0f}M filas)", flush=True)

    if not parts:
        return pd.DataFrame(columns=["msno"] + SUM_COLS + ["max_date"])
    combined = pd.concat(parts, ignore_index=True)
    result = combined.groupby("msno")[SUM_COLS].sum()
    result["max_date"] = combined.groupby("msno")["max_date"].max()
    return result.reset_index()


LOG_AGG_CACHE     = ROOT / "data" / "processed" / "user_logs_agg.parquet"
LOG_AGG_CACHE_MAR = ROOT / "data" / "processed" / "user_logs_agg_mar.parquet"

MAX_DATE_MAR      = 20170331
RECENT_CUTOFF_MAR = 20170228  # ~30 days before 2017-03-31


def load_log_features(use_cache: bool = True, max_date: int = MAX_DATE, recent_cutoff: int = RECENT_CUTOFF) -> pd.DataFrame:
    cache_path = LOG_AGG_CACHE_MAR if max_date > MAX_DATE else LOG_AGG_CACHE
    if use_cache and cache_path.exists():
        print(f"Cargando caché de user_logs ({cache_path.name})...")
        return pd.read_parquet(cache_path)

    SUM_COLS = [
        "n_days", "total_secs_sum", "num_100_sum", "num_unq_sum",
        "total_songs_sum", "recent_n_days", "recent_total_secs_sum",
    ]

    print("Agregando user_logs.csv...")
    agg1 = aggregate_logs_chunked(DATA_RAW / "user_logs.csv", max_date, recent_cutoff)
    print(f"  → {len(agg1):,} usuarios")

    print("Agregando user_logs_v2.csv...")
    agg2 = aggregate_logs_chunked(DATA_RAW / "user_logs_v2.csv", max_date, recent_cutoff)
    print(f"  → {len(agg2):,} usuarios")

    non_empty = [df for df in [agg1, agg2] if len(df) > 0]
    if len(non_empty) == 1:
        print(f"  → user_logs_v2 sin datos válidos (post-cutoff), usando solo user_logs")
        result = non_empty[0]
    else:
        print("Combinando ambos archivos...")
        combined = pd.concat(non_empty, ignore_index=True)
        result = combined.groupby("msno")[SUM_COLS].sum()
        result["max_date"] = combined.groupby("msno")["max_date"].max()
        result = result.reset_index()
        print(f"  → {len(result):,} usuarios únicos en total")

    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False)
    print(f"  Caché guardado en {cache_path.name}")
    return result


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def build_log_features(df: pd.DataFrame, prediction_date: pd.Timestamp = PREDICTION_DATE) -> pd.DataFrame:
    df = df.copy()
    df["avg_daily_secs"] = df["total_secs_sum"] / df["n_days"]
    df["avg_daily_completed"] = df["num_100_sum"] / df["n_days"]
    df["avg_daily_unq"] = df["num_unq_sum"] / df["n_days"]
    df["completion_ratio"] = np.where(
        df["total_songs_sum"] > 0,
        df["num_100_sum"] / df["total_songs_sum"],
        np.nan,
    )
    df["last_date"] = pd.to_datetime(
        df["max_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["days_since_last"] = (prediction_date - df["last_date"]).dt.days

    overall_daily = df["total_secs_sum"] / df["n_days"].clip(lower=1)
    recent_daily = df["recent_total_secs_sum"] / df["recent_n_days"].clip(lower=1)
    df["listening_trend"] = recent_daily - overall_daily

    return df


# ── ANÁLISIS ─────────────────────────────────────────────────────────────────

def analyze_logs(df: pd.DataFrame, labels: pd.DataFrame) -> dict:
    merged = df.merge(labels, on="msno", how="inner")
    global_rate = merged["is_churn"].mean()
    print(f"Usuarios en análisis: {len(merged):,} | churn rate: {global_rate:.2%}")

    FEATURE_COLS = [
        "n_days", "avg_daily_secs", "avg_daily_completed",
        "avg_daily_unq", "completion_ratio", "days_since_last",
        "listening_trend",
    ]
    summary = merged.groupby("is_churn")[FEATURE_COLS].mean().T
    summary.columns = ["Renewal (0)", "Churn (1)"]
    summary["diff_%"] = (
        (summary["Churn (1)"] - summary["Renewal (0)"]) / summary["Renewal (0)"].abs() * 100
    ).round(1)

    print("\n" + "═" * 60)
    print("USER LOGS — Churn vs Renewal")
    print(summary.to_string())

    recency_buckets = pd.cut(
        merged["days_since_last"].clip(upper=365),
        bins=[0, 7, 30, 60, 90, 180, 365],
        labels=["<1w", "1w-1m", "1-2m", "2-3m", "3-6m", "6m+"],
    )
    recency_churn = (
        merged.assign(recency_bucket=recency_buckets)
        .dropna(subset=["recency_bucket"])
        .groupby("recency_bucket", observed=True)["is_churn"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
    )
    print("\n── Churn rate por recencia del último log ──")
    print(recency_churn.to_string())

    return {"merged": merged, "summary": summary, "recency_churn": recency_churn}


# ── PLOTS ──────────────────────────────────────────────────────────────────────

def plot_listening_vs_churn(merged: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    features = [
        ("n_days", "Días con actividad de escucha"),
        ("avg_daily_secs", "Segundos escuchados / día"),
        ("avg_daily_completed", "Canciones completadas / día"),
        ("avg_daily_unq", "Canciones únicas / día"),
        ("completion_ratio", "Tasa de completitud (num_100 / total)"),
        ("days_since_last", "Días desde el último log"),
    ]
    for ax, (col, title) in zip(axes.flat, features):
        lo, hi = merged[col].quantile([0.01, 0.99])
        data = merged[merged[col].between(lo, hi)]
        renewal = data[data["is_churn"] == 0][col].dropna()
        churn = data[data["is_churn"] == 1][col].dropna()
        ax.hist(renewal, bins=50, alpha=0.6, color="#4CAF50", label="Renewal", density=True)
        ax.hist(churn, bins=50, alpha=0.6, color="#F44336", label="Churn", density=True)
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.suptitle("Comportamiento de escucha: Churn vs Renewal", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES / "ul_churn_vs_renewal.png", dpi=150)
    plt.show()


def plot_listening_trend(merged: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    lo, hi = merged["listening_trend"].quantile([0.01, 0.99])
    data = merged[merged["listening_trend"].between(lo, hi)]
    ax.hist(
        data[data["is_churn"] == 0]["listening_trend"],
        bins=60, alpha=0.6, color="#4CAF50", label="Renewal", density=True,
    )
    ax.hist(
        data[data["is_churn"] == 1]["listening_trend"],
        bins=60, alpha=0.6, color="#F44336", label="Churn", density=True,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Tendencia de escucha: últimos 30 días vs promedio histórico")
    ax.set_xlabel("Δ segundos/día (reciente − histórico)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "ul_listening_trend.png", dpi=150)
    plt.show()


def plot_churn_by_recency(results: dict) -> None:
    rc = results["recency_churn"]
    global_rate = results["merged"]["is_churn"].mean()
    colors = ["#F44336" if v > global_rate else "#4CAF50" for v in rc["churn_rate"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rc.index.astype(str), rc["churn_rate"], color=colors, edgecolor="white")
    ax.axhline(global_rate, color="black", linestyle="--", label=f"Global {global_rate:.1%}")
    for i, (rate, n) in enumerate(zip(rc["churn_rate"], rc["n"])):
        ax.text(i, rate + 0.002, f"n={n:,}", ha="center", fontsize=8)
    ax.set_title("Churn rate por recencia del último log de escucha")
    ax.set_xlabel("Días desde último log")
    ax.set_ylabel("Churn rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "ul_churn_by_recency.png", dpi=150)
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    raw_agg = load_log_features()
    print("\nDerivando features por usuario...")
    df_features = build_log_features(raw_agg)

    print("\nCargando etiquetas de train...")
    labels = pd.read_csv(DATA_RAW / "train.csv")[["msno", "is_churn"]]

    print("\nAnalizando comportamiento de escucha vs churn...")
    results = analyze_logs(df_features, labels)

    print("\nGenerando plots...")
    plot_listening_vs_churn(results["merged"])
    plot_listening_trend(results["merged"])
    plot_churn_by_recency(results)

    print("\n✓ EDA user_logs completo. Figuras en reports/figures/")
    return df_features, results


if __name__ == "__main__":
    main()
