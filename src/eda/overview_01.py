"""
EDA — Paso 1: Overview general de todos los archivos.
Cubre: shapes, dtypes, nulos, balance de clases, cobertura de keys entre tablas.
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

sns.set_theme(style="whitegrid")


# ── LOADERS ──────────────────────────────────────────────────────────────────

def load_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_RAW / "train.csv")
    train_v2 = pd.read_csv(DATA_RAW / "train_v2.csv")
    return train, train_v2


def load_members() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "members_v3.csv")


def load_transactions() -> pd.DataFrame:
    t1 = pd.read_csv(DATA_RAW / "transactions.csv")
    t2 = pd.read_csv(DATA_RAW / "transactions_v2.csv")
    return pd.concat([t1, t2], ignore_index=True)


def load_user_logs_sample(nrows: int = 500_000) -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "user_logs.csv", nrows=nrows)


# ── ANÁLISIS ─────────────────────────────────────────────────────────────────

def analyze_train(train: pd.DataFrame, train_v2: pd.DataFrame) -> dict:
    overlap = len(set(train["msno"]) & set(train_v2["msno"]))
    results = {
        "train_shape": train.shape,
        "train_v2_shape": train_v2.shape,
        "train_nulls": train.isnull().sum().to_dict(),
        "train_churn_rate": train["is_churn"].mean(),
        "train_v2_churn_rate": train_v2["is_churn"].mean(),
        "user_overlap": overlap,
        "train_unique_users": train["msno"].nunique(),
        "train_v2_unique_users": train_v2["msno"].nunique(),
    }
    print("═" * 50)
    print("TRAIN")
    print(f"  train shape      : {results['train_shape']}")
    print(f"  train_v2 shape   : {results['train_v2_shape']}")
    print(f"  churn rate train : {results['train_churn_rate']:.2%}")
    print(f"  churn rate v2    : {results['train_v2_churn_rate']:.2%}")
    print(f"  overlap usuarios : {overlap}")
    return results


def analyze_members(members: pd.DataFrame) -> dict:
    valid_age = members[(members["bd"] > 0) & (members["bd"] < 100)]["bd"]
    results = {
        "shape": members.shape,
        "nulls": members.isnull().sum().to_dict(),
        "dtypes": members.dtypes.astype(str).to_dict(),
        "age_valid_pct": len(valid_age) / len(members),
        "age_stats": valid_age.describe().to_dict(),
        "gender_dist": members["gender"].value_counts(dropna=False).to_dict(),
        "city_top10": members["city"].value_counts().head(10).to_dict(),
        "registered_via": members["registered_via"].value_counts().to_dict(),
    }
    print("═" * 50)
    print("MEMBERS")
    print(f"  shape            : {results['shape']}")
    print(f"  nulos:\n{members.isnull().sum()}")
    print(f"  edad válida (<100, >0): {results['age_valid_pct']:.2%} de filas")
    print(f"  género:\n{members['gender'].value_counts(dropna=False)}")
    return results


def analyze_transactions(transactions: pd.DataFrame) -> dict:
    results = {
        "shape": transactions.shape,
        "nulls": transactions.isnull().sum().to_dict(),
        "auto_renew_rate": transactions["is_auto_renew"].mean(),
        "cancel_rate": transactions["is_cancel"].mean(),
        "payment_methods": transactions["payment_method_id"].value_counts().head(10).to_dict(),
        "plan_price_stats": transactions["plan_list_price"].describe().to_dict(),
        "unique_users": transactions["msno"].nunique(),
    }
    print("═" * 50)
    print("TRANSACTIONS")
    print(f"  shape            : {results['shape']}")
    print(f"  usuarios únicos  : {results['unique_users']:,}")
    print(f"  auto-renew rate  : {results['auto_renew_rate']:.2%}")
    print(f"  cancel rate      : {results['cancel_rate']:.2%}")
    print(f"  nulos:\n{transactions.isnull().sum()}")
    return results


def analyze_user_logs(logs: pd.DataFrame) -> dict:
    results = {
        "shape": logs.shape,
        "nulls": logs.isnull().sum().to_dict(),
        "stats": logs.describe().to_dict(),
        "unique_users": logs["msno"].nunique(),
        "date_range": (logs["date"].min(), logs["date"].max()),
    }
    print("═" * 50)
    print("USER LOGS (muestra)")
    print(f"  shape            : {results['shape']}")
    print(f"  usuarios únicos  : {results['unique_users']:,}")
    print(f"  rango de fechas  : {results['date_range']}")
    print(f"  nulos:\n{logs.isnull().sum()}")
    return results


def analyze_key_coverage(
    train: pd.DataFrame,
    members: pd.DataFrame,
    transactions: pd.DataFrame,
) -> dict:
    train_users = set(train["msno"])
    results = {
        "train_in_members": len(train_users & set(members["msno"])) / len(train_users),
        "train_in_transactions": len(train_users & set(transactions["msno"])) / len(train_users),
    }
    print("═" * 50)
    print("COBERTURA DE KEYS")
    print(f"  train users en members     : {results['train_in_members']:.2%}")
    print(f"  train users en transactions: {results['train_in_transactions']:.2%}")
    return results


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_churn_balance(train: pd.DataFrame, train_v2: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, df, title in zip(
        axes,
        [train, train_v2],
        ["Train (Feb 2017)", "Train v2 (Mar 2017)"],
    ):
        counts = df["is_churn"].value_counts()
        ax.pie(
            counts,
            labels=["Renewal", "Churn"],
            autopct="%1.1f%%",
            colors=["#4CAF50", "#F44336"],
            startangle=90,
        )
        ax.set_title(title)
    plt.suptitle("Distribución de Churn")
    plt.tight_layout()
    plt.savefig(FIGURES / "churn_balance.png", dpi=150)
    plt.show()


def plot_age_distribution(members: pd.DataFrame) -> None:
    valid = members[(members["bd"] > 0) & (members["bd"] < 100)]["bd"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(members["bd"], bins=50, color="#2196F3", edgecolor="white")
    axes[0].set_title("Edad completa (con outliers)")
    axes[0].set_xlabel("bd")
    axes[1].hist(valid, bins=50, color="#4CAF50", edgecolor="white")
    axes[1].set_title("Edad filtrada (0 < bd < 100)")
    axes[1].set_xlabel("bd")
    plt.suptitle("Distribución de Edad — members_v3")
    plt.tight_layout()
    plt.savefig(FIGURES / "age_distribution.png", dpi=150)
    plt.show()


def plot_transaction_prices(transactions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    valid_price = transactions[transactions["plan_list_price"] > 0]["plan_list_price"]
    axes[0].hist(valid_price.clip(upper=500), bins=50, color="#FF9800", edgecolor="white")
    axes[0].set_title("Precio del plan (NTD, recortado a 500)")
    axes[0].set_xlabel("plan_list_price")
    top_methods = transactions["payment_method_id"].value_counts().head(10)
    axes[1].barh(top_methods.index.astype(str), top_methods.values, color="#9C27B0")
    axes[1].set_title("Top 10 métodos de pago")
    axes[1].set_xlabel("Transacciones")
    plt.suptitle("Transacciones — overview")
    plt.tight_layout()
    plt.savefig(FIGURES / "transactions_overview.png", dpi=150)
    plt.show()


def plot_listening_behavior(logs: pd.DataFrame) -> None:
    cols = ["num_25", "num_50", "num_75", "num_985", "num_100"]
    medians = logs[cols].median()
    fig, ax = plt.subplots(figsize=(8, 4))
    medians.plot(kind="bar", ax=ax, color="#00BCD4", edgecolor="white")
    ax.set_title("Mediana de canciones por nivel de escucha (muestra)")
    ax.set_xlabel("Nivel de reproducción")
    ax.set_ylabel("Canciones (mediana)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "listening_behavior.png", dpi=150)
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Cargando train...")
    train, train_v2 = load_train()
    train_results = analyze_train(train, train_v2)

    print("\nCargando members...")
    members = load_members()
    members_results = analyze_members(members)

    print("\nCargando transactions...")
    transactions = load_transactions()
    tx_results = analyze_transactions(transactions)

    print("\nCargando user_logs (500K filas)...")
    logs = load_user_logs_sample()
    logs_results = analyze_user_logs(logs)

    print("\nAnalizando cobertura de keys...")
    coverage = analyze_key_coverage(train, members, transactions)

    print("\nGenerando plots...")
    plot_churn_balance(train, train_v2)
    plot_age_distribution(members)
    plot_transaction_prices(transactions)
    plot_listening_behavior(logs)

    print("\n✓ EDA overview completo. Figuras guardadas en reports/figures/")
    return {
        "train": train_results,
        "members": members_results,
        "transactions": tx_results,
        "logs": logs_results,
        "coverage": coverage,
    }


if __name__ == "__main__":
    main()
