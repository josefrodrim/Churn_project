"""
Análisis de errores: falsos negativos y falsos positivos del mejor modelo.
Entrada : data/processed/features_train.parquet + models/best_model.joblib
Salidas : reports/figures/error_*.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.train_06 import load_data, best_f1_threshold, FEATURE_COLS, MODELS_DIR, FIGURES_DIR

RANDOM_STATE = 42


# ── CARGA ─────────────────────────────────────────────────────────────────────

def load_labeled_test() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(ROOT / "data" / "processed" / "features_train.parquet")
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["is_churn"].values

    _, X_test, _, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    df_test = df.iloc[idx_test].copy().reset_index(drop=True)
    return df_test, X_test, y_test


# ── CLASIFICACIÓN DE ERRORES ──────────────────────────────────────────────────

def classify_predictions(
    df_test: pd.DataFrame,
    proba: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    df = df_test.copy()
    df["proba_churn"] = proba
    df["pred"] = (proba >= threshold).astype(int)
    df["true"] = y_test

    df["result"] = "TN"
    df.loc[(df["pred"] == 1) & (df["true"] == 1), "result"] = "TP"
    df.loc[(df["pred"] == 1) & (df["true"] == 0), "result"] = "FP"
    df.loc[(df["pred"] == 0) & (df["true"] == 1), "result"] = "FN"
    return df


# ── ANÁLISIS ─────────────────────────────────────────────────────────────────

def analyze_errors(df: pd.DataFrame) -> None:
    counts = df["result"].value_counts()
    print("═" * 55)
    print("MATRIZ DE CONFUSIÓN")
    print(f"  TP (detectados)      : {counts.get('TP', 0):>7,}")
    print(f"  TN (correcto renewal): {counts.get('TN', 0):>7,}")
    print(f"  FP (falso alarma)    : {counts.get('FP', 0):>7,}")
    print(f"  FN (churners perdidos): {counts.get('FN', 0):>7,}")

    fn = df[df["result"] == "FN"]
    fp = df[df["result"] == "FP"]
    tp = df[df["result"] == "TP"]

    print(f"\n── Falsos negativos (FN={len(fn):,}) — churners que no detectamos ──")
    print(f"  proba_churn media : {fn['proba_churn'].mean():.3f}  (umbral={df['proba_churn'].quantile(0.5):.3f})")
    print(f"  last_is_cancel    : {fn['last_is_cancel'].mean():.3f}  vs TP {tp['last_is_cancel'].mean():.3f}")
    print(f"  last_is_auto_renew: {fn['last_is_auto_renew'].mean():.3f}  vs TP {tp['last_is_auto_renew'].mean():.3f}")
    print(f"  days_since_last   : {fn['days_since_last'].mean():.1f}   vs TP {tp['days_since_last'].mean():.1f}")
    print(f"  n_transactions    : {fn['n_transactions'].mean():.1f}   vs TP {tp['n_transactions'].mean():.1f}")

    print(f"\n── Falsos positivos (FP={len(fp):,}) — renewals mal clasificados ──")
    print(f"  proba_churn media : {fp['proba_churn'].mean():.3f}")
    print(f"  last_is_cancel    : {fp['last_is_cancel'].mean():.3f}")
    print(f"  last_is_auto_renew: {fp['last_is_auto_renew'].mean():.3f}")
    print(f"  days_since_last   : {fp['days_since_last'].mean():.1f}")


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_error_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    features = [
        ("proba_churn",       "Score P(churn)"),
        ("last_is_cancel",    "Última tx: cancelación"),
        ("last_is_auto_renew","Última tx: auto-renew"),
        ("days_since_last",   "Días desde último log"),
        ("n_transactions",    "N° transacciones"),
        ("tenure_days",       "Antigüedad (días)"),
    ]
    palette = {"TP": "#4CAF50", "FN": "#FF9800", "FP": "#F44336", "TN": "#9E9E9E"}

    for ax, (col, title) in zip(axes.flat, features):
        lo, hi = df[col].quantile([0.01, 0.99])
        for result in ["TP", "FN", "FP"]:
            subset = df[(df["result"] == result) & df[col].between(lo, hi)][col]
            if len(subset) > 0:
                ax.hist(subset, bins=40, alpha=0.55, density=True,
                        color=palette[result], label=result)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle("Distribución de features por tipo de predicción (TP / FN / FP)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_distributions.png", dpi=150)
    plt.show()


def plot_score_by_result(df: pd.DataFrame, threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribución de scores por resultado
    for result, color in [("TP", "#4CAF50"), ("FN", "#FF9800"), ("FP", "#F44336")]:
        subset = df[df["result"] == result]["proba_churn"]
        axes[0].hist(subset, bins=50, alpha=0.6, density=True,
                     color=color, label=f"{result} (n={len(subset):,})")
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"Umbral={threshold:.2f}")
    axes[0].set_title("Distribución de scores por resultado")
    axes[0].set_xlabel("P(churn)")
    axes[0].legend()

    # Score medio de FN por bucket de n_transactions
    fn = df[df["result"] == "FN"].copy()
    fn["tx_bucket"] = pd.cut(fn["n_transactions"],
                             bins=[0, 5, 10, 20, 50, 200],
                             labels=["1-5", "6-10", "11-20", "21-50", "51+"])
    fn_tx = fn.groupby("tx_bucket", observed=True)["proba_churn"].agg(["mean", "count"])
    axes[1].bar(fn_tx.index.astype(str), fn_tx["mean"],
                color="#FF9800", edgecolor="white")
    for i, (rate, n) in enumerate(zip(fn_tx["mean"], fn_tx["count"])):
        axes[1].text(i, rate + 0.003, f"n={n:,}", ha="center", fontsize=8)
    axes[1].set_title("Score medio de FN por N° transacciones")
    axes[1].set_xlabel("N° transacciones")
    axes[1].set_ylabel("P(churn) media")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_scores.png", dpi=150)
    plt.show()


def plot_confusion_heatmap(y_test: np.ndarray, preds: np.ndarray) -> None:
    cm = confusion_matrix(y_test, preds)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        [".0f", ".2%"],
        ["Conteo absoluto", "Porcentaje por fila (recall)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["Renewal", "Churn"],
                    yticklabels=["Renewal", "Churn"],
                    ax=ax, cbar=False)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(title)
    plt.suptitle("Matriz de confusión — LightGBM (test set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_confusion_matrix.png", dpi=150)
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("── Cargando datos y modelo ──")
    df_test, X_test, y_test = load_labeled_test()

    artifact = joblib.load(MODELS_DIR / "best_model.joblib")
    model = artifact["model"]
    print(f"  Modelo: {artifact['name']} | test size: {len(df_test):,}")

    proba = model.predict_proba(X_test)[:, 1]
    threshold = best_f1_threshold(y_test, proba)
    preds = (proba >= threshold).astype(int)
    print(f"  Umbral F1 óptimo: {threshold:.3f}")

    print("\n── Clasificando predicciones ──")
    df_labeled = classify_predictions(df_test, proba, y_test, threshold)
    analyze_errors(df_labeled)

    print("\n── Generando plots ──")
    plot_confusion_heatmap(y_test, preds)
    plot_error_distributions(df_labeled)
    plot_score_by_result(df_labeled, threshold)

    print("\n✓ Análisis de errores completo. Figuras en reports/figures/")
    return df_labeled


if __name__ == "__main__":
    main()
