"""
EDA — Paso 2: Análisis profundo de transacciones.
Cubre: precios, planes, descuentos, cancelaciones y comparación churn vs renewal.
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

def load_transactions() -> pd.DataFrame:
    t1 = pd.read_csv(DATA_RAW / "transactions.csv")
    t2 = pd.read_csv(DATA_RAW / "transactions_v2.csv")
    tx = pd.concat([t1, t2], ignore_index=True)
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"].astype(str), format="%Y%m%d")
    tx["membership_expire_date"] = pd.to_datetime(
        tx["membership_expire_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    return tx


def load_train_labels() -> pd.DataFrame:
    train = pd.read_csv(DATA_RAW / "train.csv")
    return train[["msno", "is_churn"]]


# ── ANÁLISIS GENERAL ─────────────────────────────────────────────────────────

def analyze_tx_patterns(tx: pd.DataFrame) -> dict:
    tx["discount"] = tx["plan_list_price"] - tx["actual_amount_paid"]
    tx["discount_pct"] = np.where(
        tx["plan_list_price"] > 0,
        tx["discount"] / tx["plan_list_price"],
        np.nan,
    )

    results = {
        "shape": tx.shape,
        "unique_users": tx["msno"].nunique(),
        "date_range": (tx["transaction_date"].min(), tx["transaction_date"].max()),
        "plan_days_counts": tx["payment_plan_days"].value_counts().head(10).to_dict(),
        "price_stats": tx["plan_list_price"].describe().to_dict(),
        "paid_stats": tx["actual_amount_paid"].describe().to_dict(),
        "discount_pct_mean": tx["discount_pct"].mean(),
        "zero_price_pct": (tx["plan_list_price"] == 0).mean(),
        "auto_renew_rate": tx["is_auto_renew"].mean(),
        "cancel_rate": tx["is_cancel"].mean(),
    }

    print("═" * 55)
    print("TRANSACTIONS — overview")
    print(f"  shape            : {results['shape']}")
    print(f"  usuarios únicos  : {results['unique_users']:,}")
    print(f"  rango fechas     : {results['date_range'][0].date()} → {results['date_range'][1].date()}")
    print(f"  descuento medio  : {results['discount_pct_mean']:.1%}")
    print(f"  precio=0 (free)  : {results['zero_price_pct']:.1%}")
    print(f"  auto-renew rate  : {results['auto_renew_rate']:.1%}")
    print(f"  cancel rate      : {results['cancel_rate']:.1%}")
    print(f"\n  Distribución payment_plan_days (top 10):")
    for days, cnt in sorted(results["plan_days_counts"].items()):
        print(f"    {days:>4} días : {cnt:>10,}")
    return results


def build_user_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Agrega transacciones a nivel usuario. Base para feature engineering."""
    tx = tx.copy()
    tx["discount_pct"] = np.where(
        tx["plan_list_price"] > 0,
        (tx["plan_list_price"] - tx["actual_amount_paid"]) / tx["plan_list_price"],
        np.nan,
    )

    tx_sorted = tx.sort_values("transaction_date")
    last = tx_sorted.groupby("msno").last().reset_index()
    first = tx_sorted.groupby("msno").first().reset_index()

    agg = tx.groupby("msno").agg(
        n_transactions=("msno", "count"),
        n_cancels=("is_cancel", "sum"),
        ever_canceled=("is_cancel", "max"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_plan_days=("payment_plan_days", "mean"),
        avg_price=("actual_amount_paid", "mean"),
        n_unique_plans=("payment_plan_days", "nunique"),
        n_payment_methods=("payment_method_id", "nunique"),
    ).reset_index()

    agg["last_is_cancel"] = last.set_index("msno")["is_cancel"].reindex(agg["msno"]).values
    agg["last_is_auto_renew"] = last.set_index("msno")["is_auto_renew"].reindex(agg["msno"]).values
    agg["last_plan_days"] = last.set_index("msno")["payment_plan_days"].reindex(agg["msno"]).values
    agg["last_price"] = last.set_index("msno")["actual_amount_paid"].reindex(agg["msno"]).values
    agg["last_list_price"] = last.set_index("msno")["plan_list_price"].reindex(agg["msno"]).values
    agg["price_trend"] = (
        last.set_index("msno")["actual_amount_paid"].reindex(agg["msno"]).values
        - first.set_index("msno")["actual_amount_paid"].reindex(agg["msno"]).values
    )
    agg["last_payment_method"] = last.set_index("msno")["payment_method_id"].reindex(agg["msno"]).values

    return agg


# ── ANÁLISIS CHURN vs RENEWAL ─────────────────────────────────────────────────

def analyze_churn_vs_renewal(user_features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    df = user_features.merge(labels, on="msno", how="inner")

    print("═" * 55)
    print("CHURN vs RENEWAL — diferencias en features de transacciones")
    print(f"  usuarios en análisis: {len(df):,}")
    print(f"  churn rate          : {df['is_churn'].mean():.2%}\n")

    numeric_cols = [
        "n_transactions", "n_cancels", "ever_canceled",
        "avg_discount_pct", "avg_plan_days", "avg_price",
        "last_is_cancel", "last_is_auto_renew", "last_plan_days",
        "last_price", "price_trend",
    ]

    summary = df.groupby("is_churn")[numeric_cols].mean().T
    summary.columns = ["Renewal (0)", "Churn (1)"]
    summary["diff_%"] = ((summary["Churn (1)"] - summary["Renewal (0)"]) / summary["Renewal (0)"].abs() * 100).round(1)
    print(summary.to_string())
    return df


# ── PLOTS ──────────────────────────────────────────────────────────────────────

def plot_plan_days_distribution(tx: pd.DataFrame) -> None:
    plan_counts = tx["payment_plan_days"].value_counts().sort_index()
    top = plan_counts[plan_counts > 10_000]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(top.index.astype(str), top.values, color="#00BCD4", edgecolor="white")
    ax.set_title("Duración de planes — transacciones con >10K ocurrencias")
    ax.set_xlabel("Días del plan")
    ax.set_ylabel("Transacciones")
    plt.tight_layout()
    plt.savefig(FIGURES / "tx_plan_days.png", dpi=150)
    plt.show()


def plot_price_distribution(tx: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    valid = tx[tx["plan_list_price"] > 0]["plan_list_price"]
    axes[0].hist(valid.clip(upper=500), bins=60, color="#FF9800", edgecolor="white")
    axes[0].set_title("plan_list_price (>0, recortado a 500 NTD)")
    axes[0].set_xlabel("NTD")

    paid = tx[tx["actual_amount_paid"] > 0]["actual_amount_paid"]
    axes[1].hist(paid.clip(upper=500), bins=60, color="#4CAF50", edgecolor="white")
    axes[1].set_title("actual_amount_paid (>0, recortado a 500 NTD)")
    axes[1].set_xlabel("NTD")

    plt.suptitle("Distribución de precios — Transactions")
    plt.tight_layout()
    plt.savefig(FIGURES / "tx_price_distribution.png", dpi=150)
    plt.show()


def plot_churn_comparisons(df_labeled: pd.DataFrame) -> None:
    """df_labeled = user_features merged con labels (is_churn)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    palette = {0: "#4CAF50", 1: "#F44336"}

    features = [
        ("n_transactions", "N° de transacciones"),
        ("n_cancels", "N° de cancelaciones"),
        ("avg_discount_pct", "Descuento promedio"),
        ("last_plan_days", "Días del último plan"),
        ("last_price", "Precio del último pago (NTD)"),
        ("last_is_auto_renew", "Auto-renew en última tx"),
    ]

    for ax, (col, title) in zip(axes.flat, features):
        clip_val = df_labeled[col].quantile(0.99)
        plot_data = df_labeled.copy()
        plot_data[col] = plot_data[col].clip(upper=clip_val)

        churn_vals = plot_data[plot_data["is_churn"] == 1][col].dropna()
        renewal_vals = plot_data[plot_data["is_churn"] == 0][col].dropna()

        ax.hist(renewal_vals, bins=40, alpha=0.6, color="#4CAF50", label="Renewal", density=True)
        ax.hist(churn_vals, bins=40, alpha=0.6, color="#F44336", label="Churn", density=True)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle("Distribuciones por grupo: Churn vs Renewal", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES / "tx_churn_vs_renewal.png", dpi=150)
    plt.show()


def plot_cancel_and_autorenew_churn(df_labeled: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, title in zip(
        axes,
        ["ever_canceled", "last_is_auto_renew"],
        ["¿Alguna vez canceló?", "Auto-renew en última transacción"],
    ):
        ct = df_labeled.groupby([col, "is_churn"]).size().unstack(fill_value=0)
        ct_pct = ct.div(ct.sum(axis=1), axis=0)
        ct_pct.plot(
            kind="bar", ax=ax,
            color=["#4CAF50", "#F44336"],
            edgecolor="white",
        )
        ax.set_title(title)
        ax.set_ylabel("Proporción")
        ax.set_xlabel("")
        ax.set_xticklabels(["No", "Sí"], rotation=0)
        ax.legend(["Renewal", "Churn"])

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1%", fontsize=8)

    plt.suptitle("Churn rate por comportamiento de cancelación / auto-renew")
    plt.tight_layout()
    plt.savefig(FIGURES / "tx_cancel_autorenew_churn.png", dpi=150)
    plt.show()


def plot_price_trend_churn(df_labeled: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    clip = df_labeled["price_trend"].quantile([0.01, 0.99]).values
    data = df_labeled[df_labeled["price_trend"].between(clip[0], clip[1])]

    renewal = data[data["is_churn"] == 0]["price_trend"]
    churn = data[data["is_churn"] == 1]["price_trend"]

    ax.hist(renewal, bins=50, alpha=0.6, color="#4CAF50", label="Renewal", density=True)
    ax.hist(churn, bins=50, alpha=0.6, color="#F44336", label="Churn", density=True)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Tendencia de precio (último - primer pago): Churn vs Renewal")
    ax.set_xlabel("NTD")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "tx_price_trend_churn.png", dpi=150)
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("Cargando transacciones...")
    tx = load_transactions()

    print("Analizando patrones generales...")
    analyze_tx_patterns(tx)

    print("\nGenerando features por usuario...")
    user_feats = build_user_features(tx)
    print(f"  features shape: {user_feats.shape}")

    print("\nCargando etiquetas de train...")
    labels = load_train_labels()

    print("\nComparando churn vs renewal...")
    df_labeled = analyze_churn_vs_renewal(user_feats, labels)

    print("\nGenerando plots...")
    plot_plan_days_distribution(tx)
    plot_price_distribution(tx)
    plot_churn_comparisons(df_labeled)
    plot_cancel_and_autorenew_churn(df_labeled)
    plot_price_trend_churn(df_labeled)

    print("\n✓ EDA transactions completo. Figuras en reports/figures/")
    return df_labeled


if __name__ == "__main__":
    main()
