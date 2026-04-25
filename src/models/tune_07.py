"""
Hyperparameter tuning: Optuna sobre LightGBM.
Entrada : data/processed/features_train.parquet
Salidas : models/lgbm_tuned.joblib
          reports/figures/tuning_history.png
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.train_06 import (
    load_data, evaluate_on_test, plot_roc_pr,
    FEATURE_COLS, MODELS_DIR, FIGURES_DIR,
)

N_TRIALS   = 60
N_FOLDS    = 3          # 3-fold para velocidad en tuning
RANDOM_STATE = 42


# ── OBJECTIVE ─────────────────────────────────────────────────────────────────

def make_objective(X_train, y_train, pos_weight):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 31, 255),
            "max_depth":         trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "is_unbalance": True,
            "random_state": RANDOM_STATE,
            "verbosity": -1,
        }

        aucs = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model = LGBMClassifier(**params)
            model.fit(X_train[train_idx], y_train[train_idx])
            proba = model.predict_proba(X_train[val_idx])[:, 1]
            aucs.append(roc_auc_score(y_train[val_idx], proba))
        return np.mean(aucs)

    return objective


# ── PLOTS ─────────────────────────────────────────────────────────────────────

def plot_tuning_history(study) -> None:
    trials = study.trials_dataframe()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(trials["number"], trials["value"], alpha=0.4, color="#2196F3", lw=1)
    best_so_far = trials["value"].cummax()
    axes[0].plot(trials["number"], best_so_far, color="#F44336", lw=2, label="Mejor acumulado")
    axes[0].set_title("Historial de optimización — Optuna")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("ROC-AUC (3-fold CV)")
    axes[0].legend()

    top_params = ["params_num_leaves", "params_learning_rate",
                  "params_max_depth", "params_min_child_samples"]
    top_params = [p for p in top_params if p in trials.columns]
    for i, param in enumerate(top_params[:4]):
        ax_twin = axes[1] if i == 0 else None
        name = param.replace("params_", "")
        axes[1].scatter(trials[param], trials["value"],
                        alpha=0.4, label=name, s=20)
    axes[1].set_title("Parámetros vs AUC")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tuning_history.png", dpi=150)
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("── Cargando features ──")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"\n── Optimización Optuna ({N_TRIALS} trials × {N_FOLDS}-fold CV) ──")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(
        make_objective(X_train, y_train, pos_weight),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nMejor AUC CV : {study.best_value:.5f}")
    print("Mejores parámetros:")
    for k, v in best.items():
        print(f"  {k:25s}: {v}")

    print("\n── Entrenando modelo final con mejores parámetros ──")
    tuned_model = LGBMClassifier(
        **best,
        is_unbalance=True,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    tuned_model.fit(X_train, y_train)

    print("\n── Evaluando en test set ──")
    # Comparación tuned vs baseline
    baseline = joblib.load(MODELS_DIR / "best_model.joblib")["model"]
    results = [
        evaluate_on_test("LightGBM baseline", baseline, X_test, y_test),
        evaluate_on_test("LightGBM tuned",    tuned_model, X_test, y_test),
    ]

    print("\n── Curvas ROC / PR ──")
    plot_roc_pr(results, y_test)
    plot_tuning_history(study)

    model_path = MODELS_DIR / "lgbm_tuned.joblib"
    joblib.dump(
        {"model": tuned_model, "features": FEATURE_COLS,
         "best_params": best, "best_cv_auc": study.best_value},
        model_path,
    )
    print(f"\n✓ Modelo tuneado guardado en {model_path}")
    return tuned_model, study


if __name__ == "__main__":
    main()
