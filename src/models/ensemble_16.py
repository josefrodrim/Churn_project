"""
Ensemble v1 — LightGBM + XGBoost + CatBoost + blend.

Sobre v5_raw añade dos modelos con sesgos inductivos distintos:
  - XGBoost:  crecimiento level-wise, tuneado con Optuna (15 trials)
  - CatBoost: árboles simétricos, features categóricas nativas, ordered boosting

Features: misma v5 (48) para los tres modelos.
Blend: promedio simple de probabilidades.

Salidas (4 submissions):
  submissions/submission_v7_xgb.csv
  submissions/submission_v8_catboost.csv
  submissions/submission_v9_blend_lgbm_xgb.csv
  submissions/submission_v10_blend3.csv

Modelos guardados:
  models/xgb_v1.joblib
  models/catboost_v1.cbm
  models/lgbm_ensemble_v1.joblib
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_RAW        = ROOT / "data" / "raw"
MODELS_DIR      = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
SUBMISSIONS_DIR.mkdir(exist_ok=True)

PREDICTION_DATE_MAR = pd.Timestamp("2017-03-31")
PREDICTION_DATE_APR = pd.Timestamp("2017-04-30")
LOG_CUTOFF_DATE     = pd.Timestamp("2017-03-31")

from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import (
    load_log_features, MAX_DATE_MAR, RECENT_CUTOFF_MAR, EXTRA_CUTOFFS_MAR_V2,
)
from src.features.build_features_05 import impute_missing
from src.models.train_06 import TARGET, RANDOM_STATE
from src.models.retrain_submit_13 import build_member_features, build_expiry_features, build_churn_lag
from src.models.retrain_submit_14 import (
    FEATURE_COLS_V5,
    build_multiwindow_log_features,
    build_tx_recency_features,
    join_all_v5,
)

# Columnas categóricas para CatBoost
CATEG_COLS = ["city", "registered_via", "last_payment_method", "gender_enc"]
CATEG_IDX  = [FEATURE_COLS_V5.index(c) for c in CATEG_COLS]

N_OPTUNA_TRIALS  = 15
TUNE_SAMPLE_FRAC = 0.25   # subsample para acelerar el tuning de XGB
N_CV_FOLDS       = 3


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_v5_datasets():
    """Construye features v5 para train_v2 (entrenamiento) y submission (abril)."""
    print("\n── Cargando logs multiventana ──")
    raw_agg_v2 = load_log_features(
        max_date=MAX_DATE_MAR,
        recent_cutoff=RECENT_CUTOFF_MAR,
        extra_cutoffs=EXTRA_CUTOFFS_MAR_V2,
    )
    print(f"  {len(raw_agg_v2):,} usuarios")

    print("\n── Cargando transacciones ──")
    tx         = load_transactions()
    tx_feats   = build_user_features(tx)
    tx_recency = build_tx_recency_features(tx)
    print(f"  {len(tx_feats):,} usuarios")

    print("\n── Features de entrenamiento (train_v2 / mar 2017) ──")
    labels_mar   = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", TARGET]]
    member_mar   = build_member_features(PREDICTION_DATE_MAR)
    expiry_train = build_expiry_features(tx, LOG_CUTOFF_DATE)
    churn_lag_tr = build_churn_lag(PREDICTION_DATE_MAR)

    df_train = join_all_v5(
        labels_mar, tx_feats, member_mar, raw_agg_v2,
        expiry_train, churn_lag_tr, tx_recency,
    )
    X = df_train[FEATURE_COLS_V5].values.astype(np.float32)
    y = df_train[TARGET].values
    print(f"  {len(df_train):,} filas | churn: {y.mean():.2%} | nulos: {df_train[FEATURE_COLS_V5].isnull().sum().sum()}")

    print("\n── Features de submission (abr 2017) ──")
    sub           = pd.read_csv(DATA_RAW / "sample_submission_v2.csv")[["msno"]]
    member_sub    = build_member_features(PREDICTION_DATE_APR)
    expiry_sub    = build_expiry_features(tx, LOG_CUTOFF_DATE)
    churn_lag_sub = build_churn_lag(PREDICTION_DATE_APR)

    df_sub = join_all_v5(
        sub, tx_feats, member_sub, raw_agg_v2,
        expiry_sub, churn_lag_sub, tx_recency,
    )
    X_sub = df_sub[FEATURE_COLS_V5].values.astype(np.float32)
    print(f"  {len(df_sub):,} filas | nulos: {df_sub[FEATURE_COLS_V5].isnull().sum().sum()}")

    return X, y, X_sub, df_sub["msno"].values


# ── XGBOOST ──────────────────────────────────────────────────────────────────

def tune_xgb(X, y):
    """Optuna sobre subsample de train_v2 para velocidad."""
    print(f"\n── Tuneando XGBoost ({N_OPTUNA_TRIALS} trials, {N_CV_FOLDS}-fold, {TUNE_SAMPLE_FRAC:.0%} subsample) ──")
    pos_weight = (y == 0).sum() / (y == 1).sum()

    # Subsample estratificado para acelerar
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - TUNE_SAMPLE_FRAC, random_state=RANDOM_STATE)
    tune_idx, _ = next(sss.split(X, y))
    X_tune, y_tune = X[tune_idx], y[tune_idx]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1200),
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, val_idx in skf.split(X_tune, y_tune):
            m = XGBClassifier(
                **params,
                scale_pos_weight=pos_weight,
                tree_method="hist",
                random_state=RANDOM_STATE,
                verbosity=0,
            )
            m.fit(X_tune[tr_idx], y_tune[tr_idx])
            proba = m.predict_proba(X_tune[val_idx])[:, 1]
            scores.append(log_loss(y_tune[val_idx], proba))
        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    print(f"  Mejor LogLoss CV: {study.best_value:.5f}")
    print(f"  Params: {study.best_params}")
    return study.best_params


def train_xgb(X, y, params):
    pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        **params,
        scale_pos_weight=pos_weight,
        tree_method="hist",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X, y)
    return model


# ── CATBOOST ─────────────────────────────────────────────────────────────────

def to_catboost_df(X: np.ndarray) -> pd.DataFrame:
    """CatBoost con DataFrame: columnas categóricas como int, resto float."""
    df = pd.DataFrame(X, columns=FEATURE_COLS_V5)
    for col in CATEG_COLS:
        df[col] = df[col].astype(int)
    return df


def train_catboost(X, y):
    print("\n── Entrenando CatBoost (ordered boosting, cats nativas) ──")
    df_cb = to_catboost_df(X)
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        cat_features=CATEG_COLS,  # nombres de columna cuando se pasa DataFrame
        auto_class_weights="Balanced",
        eval_metric="Logloss",
        random_seed=RANDOM_STATE,
        verbose=200,
    )
    model.fit(df_cb, y)
    return model


# ── HELPERS ──────────────────────────────────────────────────────────────────

def print_metrics(label, y_true, y_proba):
    print(f"  {label}")
    print(f"    ROC-AUC: {roc_auc_score(y_true, y_proba):.5f} | "
          f"PR-AUC: {average_precision_score(y_true, y_proba):.5f} | "
          f"LogLoss: {log_loss(y_true, y_proba):.5f}")


def save_submission(msno, proba, path):
    pd.DataFrame({"msno": msno, "is_churn": proba}).to_csv(path, index=False)
    print(f"  → {path.name}  (mean: {proba.mean():.4f} | min: {proba.min():.4f} | max: {proba.max():.4f})")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("ENSEMBLE v1 — LightGBM + XGBoost + CatBoost")
    print("═" * 60)

    X, y, X_sub, msno_sub = load_v5_datasets()

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print("\n── LightGBM (params tuneados, train completo) ──")
    lgbm_params = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")["best_params"]
    lgbm = LGBMClassifier(
        **lgbm_params, is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1
    )
    lgbm.fit(X, y)
    lgbm_sub = lgbm.predict_proba(X_sub)[:, 1]
    joblib.dump({"model": lgbm, "features": FEATURE_COLS_V5},
                MODELS_DIR / "lgbm_ensemble_v1.joblib")
    print(f"  Guardado: models/lgbm_ensemble_v1.joblib")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_params = tune_xgb(X, y)
    print("\n── Entrenando XGBoost final (train completo) ──")
    xgb = train_xgb(X, y, xgb_params)
    xgb_sub = xgb.predict_proba(X_sub)[:, 1]
    joblib.dump({"model": xgb, "features": FEATURE_COLS_V5, "params": xgb_params},
                MODELS_DIR / "xgb_v1.joblib")
    print(f"  Guardado: models/xgb_v1.joblib")
    save_submission(msno_sub, xgb_sub, SUBMISSIONS_DIR / "submission_v7_xgb.csv")

    # ── CatBoost ──────────────────────────────────────────────────────────────
    cat = train_catboost(X, y)
    df_sub_cb = to_catboost_df(X_sub)
    cat_sub   = cat.predict_proba(df_sub_cb)[:, 1]
    cat.save_model(str(MODELS_DIR / "catboost_v1.cbm"))
    print(f"\n  Guardado: models/catboost_v1.cbm")
    save_submission(msno_sub, cat_sub, SUBMISSIONS_DIR / "submission_v8_catboost.csv")

    # ── Blends ────────────────────────────────────────────────────────────────
    print("\n── Blends ──")
    blend_lx  = (lgbm_sub + xgb_sub) / 2
    blend_all = (lgbm_sub + xgb_sub + cat_sub) / 3

    save_submission(msno_sub, blend_lx,  SUBMISSIONS_DIR / "submission_v9_blend_lgbm_xgb.csv")
    save_submission(msno_sub, blend_all, SUBMISSIONS_DIR / "submission_v10_blend3.csv")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n── Predicciones submission ──")
    print(f"  {'Modelo':<22} {'Mean':>8} {'Min':>8} {'Max':>8}")
    for name, preds in [
        ("LightGBM",       lgbm_sub),
        ("XGBoost",        xgb_sub),
        ("CatBoost",       cat_sub),
        ("Blend LX",       blend_lx),
        ("Blend 3",        blend_all),
    ]:
        print(f"  {name:<22} {preds.mean():>8.4f} {preds.min():>8.4f} {preds.max():>8.4f}")

    print("\n✓ Submissions listas:")
    for fname in [
        "submission_v7_xgb.csv",
        "submission_v8_catboost.csv",
        "submission_v9_blend_lgbm_xgb.csv",
        "submission_v10_blend3.csv",
    ]:
        print(f"  submissions/{fname}")


if __name__ == "__main__":
    main()
