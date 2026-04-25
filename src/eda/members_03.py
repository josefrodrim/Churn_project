"""
EDA — Paso 3: Demografía de members cruzada con etiquetas de churn.
Cubre: ciudad, método de registro, género, edad, antigüedad vs churn rate.
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

sns.set_theme(style="whitegrid")


# ── LOADERS ──────────────────────────────────────────────────────────────────

def load_members_labeled() -> pd.DataFrame:
    members = pd.read_csv(DATA_RAW / "members_v3.csv")
    train = pd.read_csv(DATA_RAW / "train.csv")[["msno", "is_churn"]]

    members["registration_init_time"] = pd.to_datetime(
        members["registration_init_time"].astype(str), format="%Y%m%d", errors="coerce"
    )
    members["gender"] = members["gender"].fillna("unknown")
    members["bd_valid"] = members["bd"].between(1, 99)
    members["age"] = np.where(members["bd_valid"], members["bd"], np.nan)
    members["age_group"] = pd.cut(
        members["age"],
        bins=[0, 18, 25, 35, 45, 55, 100],
        labels=["<18", "18-25", "25-35", "35-45", "45-55", "55+"],
    )
    members["tenure_days"] = (PREDICTION_DATE - members["registration_init_time"]).dt.days
    members["tenure_years"] = (members["tenure_days"] / 365).round(1)
    members["tenure_bucket"] = pd.cut(
        members["tenure_days"],
        bins=[0, 90, 365, 730, 1095, 1825, 99999],
        labels=["<3m", "3-12m", "1-2y", "2-3y", "3-5y", "5y+"],
    )

    df = members.merge(train, on="msno", how="inner")
    print(f"Usuarios en análisis: {len(df):,} | churn rate: {df['is_churn'].mean():.2%}")
    return df


# ── ANÁLISIS ─────────────────────────────────────────────────────────────────

def analyze_demographics(df: pd.DataFrame) -> dict:
    global_rate = df["is_churn"].mean()

    city_churn = (
        df.groupby("city")["is_churn"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
        .query("n >= 500")
        .sort_values("churn_rate", ascending=False)
    )

    reg_churn = (
        df.groupby("registered_via")["is_churn"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
        .query("n >= 500")
        .sort_values("churn_rate", ascending=False)
    )

    gender_churn = (
        df.groupby("gender")["is_churn"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
        .sort_values("churn_rate", ascending=False)
    )

    age_churn = (
        df.dropna(subset=["age_group"])
        .groupby("age_group", observed=True)["is_churn"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
    )

    tenure_churn = (
        df.dropna(subset=["tenure_bucket"])
        .groupby("tenure_bucket", observed=True)["is_churn"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "n"})
    )

    print("═" * 55)
    print(f"CHURN RATE GLOBAL: {global_rate:.2%}\n")

    print("── Por método de registro ──")
    print(reg_churn.to_string())
    print()
    print("── Por género ──")
    print(gender_churn.to_string())
    print()
    print("── Por grupo de edad ──")
    print(age_churn.to_string())
    print()
    print("── Por antigüedad ──")
    print(tenure_churn.to_string())

    return {
        "global_rate": global_rate,
        "city_churn": city_churn,
        "reg_churn": reg_churn,
        "gender_churn": gender_churn,
        "age_churn": age_churn,
        "tenure_churn": tenure_churn,
    }


# ── PLOTS ──────────────────────────────────────────────────────────────────────

def plot_churn_by_city(results: dict) -> None:
    city = results["city_churn"].head(20).sort_values("churn_rate")
    global_rate = results["global_rate"]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#F44336" if v > global_rate else "#4CAF50" for v in city["churn_rate"]]
    ax.barh(city.index.astype(str), city["churn_rate"], color=colors, edgecolor="white")
    ax.axvline(global_rate, color="black", linestyle="--", linewidth=1, label=f"Global {global_rate:.1%}")
    ax.set_title("Churn rate por ciudad (top 20, mín. 500 usuarios)")
    ax.set_xlabel("Churn rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "mem_churn_by_city.png", dpi=150)
    plt.show()


def plot_churn_by_registration(results: dict) -> None:
    reg = results["reg_churn"].sort_values("churn_rate")
    global_rate = results["global_rate"]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#F44336" if v > global_rate else "#4CAF50" for v in reg["churn_rate"]]
    ax.barh(reg.index.astype(str), reg["churn_rate"], color=colors, edgecolor="white")
    ax.axvline(global_rate, color="black", linestyle="--", label=f"Global {global_rate:.1%}")
    for i, (rate, n) in enumerate(zip(reg["churn_rate"], reg["n"])):
        ax.text(rate + 0.001, i, f"n={n:,}", va="center", fontsize=8)
    ax.set_title("Churn rate por método de registro")
    ax.set_xlabel("Churn rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "mem_churn_by_registration.png", dpi=150)
    plt.show()


def plot_churn_by_demographics(results: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    global_rate = results["global_rate"]

    # Género
    g = results["gender_churn"].sort_values("churn_rate")
    axes[0].bar(g.index, g["churn_rate"], color=["#2196F3", "#E91E63", "#9E9E9E"][:len(g)], edgecolor="white")
    axes[0].axhline(global_rate, color="red", linestyle="--")
    axes[0].set_title("Churn rate por género")
    axes[0].set_ylabel("Churn rate")

    # Edad
    a = results["age_churn"]
    axes[1].bar(a.index.astype(str), a["churn_rate"], color="#FF9800", edgecolor="white")
    axes[1].axhline(global_rate, color="red", linestyle="--")
    axes[1].set_title("Churn rate por grupo de edad")

    # Antigüedad
    t = results["tenure_churn"]
    axes[2].bar(t.index.astype(str), t["churn_rate"], color="#9C27B0", edgecolor="white")
    axes[2].axhline(global_rate, color="red", linestyle="--")
    axes[2].set_title("Churn rate por antigüedad del usuario")

    plt.suptitle("Churn rate por perfil demográfico", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES / "mem_churn_demographics.png", dpi=150)
    plt.show()


def plot_tenure_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    churn = df[df["is_churn"] == 1]["tenure_days"].clip(upper=df["tenure_days"].quantile(0.99))
    renewal = df[df["is_churn"] == 0]["tenure_days"].clip(upper=df["tenure_days"].quantile(0.99))

    axes[0].hist(renewal, bins=60, alpha=0.6, color="#4CAF50", label="Renewal", density=True)
    axes[0].hist(churn, bins=60, alpha=0.6, color="#F44336", label="Churn", density=True)
    axes[0].set_title("Antigüedad (días): Churn vs Renewal")
    axes[0].set_xlabel("Días desde registro")
    axes[0].legend()

    city_top = df.groupby("city")["is_churn"].agg(["mean", "count"]).query("count >= 1000")
    city_top = city_top.sort_values("mean", ascending=False).head(15)
    axes[1].barh(
        city_top.index.astype(str),
        city_top["mean"],
        color=["#F44336" if v > df["is_churn"].mean() else "#4CAF50" for v in city_top["mean"]],
        edgecolor="white",
    )
    axes[1].axvline(df["is_churn"].mean(), color="black", linestyle="--")
    axes[1].set_title("Top 15 ciudades por churn rate (mín. 1K usuarios)")
    axes[1].set_xlabel("Churn rate")

    plt.tight_layout()
    plt.savefig(FIGURES / "mem_tenure_city.png", dpi=150)
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Cargando members con etiquetas...")
    df = load_members_labeled()

    print("\nAnalizando demografía vs churn...")
    results = analyze_demographics(df)

    print("\nGenerando plots...")
    plot_churn_by_city(results)
    plot_churn_by_registration(results)
    plot_churn_by_demographics(results)
    plot_tenure_distribution(df)

    print("\n✓ EDA members completo. Figuras en reports/figures/")
    return df, results


if __name__ == "__main__":
    main()
