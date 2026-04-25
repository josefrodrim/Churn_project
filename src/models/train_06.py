"""
Modelado: baseline logístico + XGBoost + LightGBM con validación cruzada.
Entrada : data/processed/features_train.parquet
Salidas : models/best_model.joblib  — mejor modelo por AUC en CV
          reports/figures/           — curvas ROC/PR, SHAP
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    f1_score, classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

FEATURES_PATH = ROOT / "data" / "processed" / "features_train.parquet"
MODELS_DIR    = ROOT / "models"
FIGURES_DIR   = ROOT / "reports" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    # transacciones
    "n_transactions", "n_cancels", "ever_canceled",
    "avg_discount_pct", "avg_plan_days", "avg_price",
    "n_unique_plans", "n_payment_methods",
    "last_is_cancel", "last_is_auto_renew",
    "last_plan_days", "last_price", "last_list_price",
    "price_trend", "last_payment_method",
    # members
    "city", "registered_via", "gender_enc",
    "age", "bd_valid", "tenure_days",
    "has_member_record",
    # user logs
    "n_days", "avg_daily_secs", "avg_daily_completed",
    "avg_daily_unq", "completion_ratio",
    "days_since_last", "listening_trend",
    "has_log_record",
]
TARGET = "is_churn"
N_FOLDS = 5
RANDOM_STATE = 42


# ── CARGA Y SPLIT ─────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET].values
    print(f"Dataset: {X.shape[0]:,} filas × {X.shape[1]} features | churn rate: {y.mean():.2%}")
    return X, y


# ── MODELOS ───────────────────────────────────────────────────────────────────

def get_models(pos_weight: float) -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }


# ── VALIDACIÓN CRUZADA ────────────────────────────────────────────────────────

def cross_validate_models(
    X: np.ndarray, y: np.ndarray, models: dict
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {name: {"auc": [], "pr_auc": []} for name in models}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_val)[:, 1]
            results[name]["auc"].append(roc_auc_score(y_val, proba))
            results[name]["pr_auc"].append(average_precision_score(y_val, proba))

        print(f"  fold {fold}/{N_FOLDS} ✓", flush=True)

    rows = []
    for name, scores in results.items():
        rows.append({
            "model": name,
            "auc_mean": np.mean(scores["auc"]),
            "auc_std":  np.std(scores["auc"]),
            "pr_auc_mean": np.mean(scores["pr_auc"]),
            "pr_auc_std":  np.std(scores["pr_auc"]),
        })
    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False)


# ── EVALUACIÓN FINAL ──────────────────────────────────────────────────────────

def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thresholds[np.argmax(f1s)])


def evaluate_on_test(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    pr    = average_precision_score(y_test, proba)
    thr   = best_f1_threshold(y_test, proba)
    preds = (proba >= thr).astype(int)
    f1    = f1_score(y_test, preds)

    print(f"\n{'═'*55}")
    print(f"TEST — {name}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  PR-AUC    : {pr:.4f}")
    print(f"  F1 (thr={thr:.2f}): {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["Renewal", "Churn"], digits=3))
    return {"name": name, "auc": auc, "pr_auc": pr, "f1": f1, "threshold": thr, "proba": proba}


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_roc_pr(results_list: list, y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for res, color in zip(results_list, colors):
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        prec, rec, _ = precision_recall_curve(y_test, res["proba"])
        label_roc = f"{res['name']} (AUC={res['auc']:.3f})"
        label_pr  = f"{res['name']} (PR-AUC={res['pr_auc']:.3f})"
        axes[0].plot(fpr, tpr, color=color, label=label_roc, lw=1.5)
        axes[1].plot(rec, prec, color=color, label=label_pr, lw=1.5)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].axhline(y_test.mean(), color="k", linestyle="--", lw=1,
                    label=f"Baseline ({y_test.mean():.3f})")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.suptitle("Curvas de evaluación — test set", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_roc_pr.png", dpi=150)
    plt.show()


def plot_shap(model, X_test: np.ndarray, n_samples: int = 5000) -> None:
    idx = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_COLS,
                      show=False, max_display=20)
    axes[0].set_title("SHAP — impacto en la predicción")

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[-20:]
    colors = ["#F44336" if mean_abs[i] > np.median(mean_abs) else "#2196F3" for i in order]
    axes[1].barh(
        [FEATURE_COLS[i] for i in order],
        mean_abs[order],
        color=colors, edgecolor="white",
    )
    axes[1].set_title("SHAP — importancia media absoluta (top 20)")
    axes[1].set_xlabel("|SHAP value|")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_shap.png", dpi=150)
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("── Cargando features ──")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | scale_pos_weight: {pos_weight:.1f}")

    models = get_models(pos_weight)

    print(f"\n── Validación cruzada ({N_FOLDS} folds) ──")
    cv_results = cross_validate_models(X_train, y_train, models)
    print("\n" + "═" * 55)
    print("RESULTADOS CV")
    print(cv_results.to_string(index=False, float_format="{:.4f}".format))

    best_name = cv_results.iloc[0]["model"]
    print(f"\n── Mejor modelo en CV: {best_name} ──")

    print("\n── Entrenando modelos finales en train completo ──")
    final_models = get_models(pos_weight)
    for name, model in final_models.items():
        model.fit(X_train, y_train)
        print(f"  {name} ✓")

    print("\n── Evaluando en test set ──")
    test_results = [evaluate_on_test(n, m, X_test, y_test) for n, m in final_models.items()]

    print("\n── Generando curvas ROC / PR ──")
    plot_roc_pr(test_results, y_test)

    best_model = final_models[best_name]
    print(f"\n── SHAP sobre {best_name} ──")
    if not isinstance(best_model, Pipeline):
        plot_shap(best_model, X_test)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump({"model": best_model, "features": FEATURE_COLS, "name": best_name}, model_path)
    print(f"\n✓ Modelo guardado en {model_path}")

    return final_models, cv_results, test_results


if __name__ == "__main__":
    main()
