"""
Fase 1 — Entrenamiento con MLflow tracking + Model Registry.

Sobre ensemble_16 añade:
  - Logging de params, métricas y artefactos por modelo en MLflow
  - Ensemble empaquetado como PyfuncModel (un solo artefacto versionado)
  - Registro en MLflow Model Registry bajo el nombre 'churn-ensemble'
  - Transición automática a Staging si pasa el quality gate (LogLoss ≤ 0.240)

Ejecutar:
  mlflow server --host 0.0.0.0 --port 5000 &   # en otra terminal
  python -m src.models.train_mlflow_17
"""

import sys
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

DATA_RAW   = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"

from src.api.config import get_settings
from src.eda.transactions_02 import load_transactions, build_user_features
from src.eda.user_logs_04 import (
    load_log_features, MAX_DATE_MAR, RECENT_CUTOFF_MAR, EXTRA_CUTOFFS_MAR_V2,
)
from src.models.train_06 import TARGET, RANDOM_STATE
from src.models.retrain_submit_13 import build_member_features, build_expiry_features, build_churn_lag
from src.models.retrain_submit_14 import (
    FEATURE_COLS_V5, build_tx_recency_features, join_all_v5,
    PREDICTION_DATE_MAR, LOG_CUTOFF_DATE,
)
from src.models.ensemble_16 import (
    CATEG_COLS, CATEG_IDX,
    tune_xgb, train_xgb, train_catboost, to_catboost_df,
)

EXPERIMENT_NAME = "churn-kkbox"
MODEL_NAME      = "churn-ensemble"
MAX_LOG_LOSS    = get_settings().max_log_loss   # 0.240


# ── ENSEMBLE PYFUNC ───────────────────────────────────────────────────────────

class ChurnEnsemble(mlflow.pyfunc.PythonModel):
    """
    Wrapper Pyfunc que encapsula los 3 modelos + blend.
    MLflow serializa esto como un único artefacto versionable.
    """

    def __init__(self, lgbm, xgb, catboost, feature_cols, categ_cols, blend_weights=None):
        self.lgbm         = lgbm
        self.xgb          = xgb
        self.catboost     = catboost
        self.feature_cols = feature_cols
        self.categ_cols   = categ_cols
        self.weights      = blend_weights or [1/3, 1/3, 1/3]

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        X = model_input[self.feature_cols].values.astype(np.float32)

        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        p_xgb  = self.xgb.predict_proba(X)[:, 1]

        df_cb = pd.DataFrame(X, columns=self.feature_cols)
        for col in self.categ_cols:
            df_cb[col] = df_cb[col].astype(int)
        p_cat = self.catboost.predict_proba(df_cb)[:, 1]

        return (
            self.weights[0] * p_lgbm
            + self.weights[1] * p_xgb
            + self.weights[2] * p_cat
        )


# ── VALIDACIÓN INTERNA ────────────────────────────────────────────────────────

def cross_val_metrics(model_fn, X, y, n_folds=3):
    """CV estratificado — devuelve métricas promedio para el quality gate."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    aucs, prs, losses = [], [], []
    for tr_idx, val_idx in skf.split(X, y):
        m = model_fn()
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], p))
        prs.append(average_precision_score(y[val_idx], p))
        losses.append(log_loss(y[val_idx], p))
    return np.mean(aucs), np.mean(prs), np.mean(losses)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("═" * 60)
    print("TRAIN + MLFLOW TRACKING — ensemble v1")
    print("═" * 60)

    # ── Datos ─────────────────────────────────────────────────────────────────
    print("\n── Cargando datos ──")
    raw_agg_v2 = load_log_features(
        max_date=MAX_DATE_MAR,
        recent_cutoff=RECENT_CUTOFF_MAR,
        extra_cutoffs=EXTRA_CUTOFFS_MAR_V2,
    )
    tx         = load_transactions()
    tx_feats   = build_user_features(tx)
    tx_recency = build_tx_recency_features(tx)

    labels_mar   = pd.read_csv(DATA_RAW / "train_v2.csv")[["msno", TARGET]]
    df_train     = join_all_v5(
        labels_mar, tx_feats,
        build_member_features(PREDICTION_DATE_MAR), raw_agg_v2,
        build_expiry_features(tx, LOG_CUTOFF_DATE),
        build_churn_lag(PREDICTION_DATE_MAR), tx_recency,
    )
    X = df_train[FEATURE_COLS_V5].values.astype(np.float32)
    y = df_train[TARGET].values
    print(f"  {len(df_train):,} filas | churn: {y.mean():.2%} | features: {len(FEATURE_COLS_V5)}")

    pos_weight = (y == 0).sum() / (y == 1).sum()

    # ── Run principal ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="ensemble-blend3") as run:
        print(f"\n── MLflow run: {run.info.run_id[:8]}... ──")

        mlflow.log_params({
            "n_features":    len(FEATURE_COLS_V5),
            "train_rows":    len(X),
            "churn_rate":    round(float(y.mean()), 4),
            "blend_weights": "1/3-1/3-1/3",
            "pos_weight":    round(pos_weight, 2),
        })

        # ── LightGBM ──────────────────────────────────────────────────────────
        print("\n── LightGBM ──")
        lgbm_params = joblib.load(MODELS_DIR / "lgbm_tuned.joblib")["best_params"]
        mlflow.log_params({f"lgbm_{k}": v for k, v in lgbm_params.items()})

        def lgbm_fn():
            return LGBMClassifier(
                **lgbm_params, is_unbalance=True,
                random_state=RANDOM_STATE, verbosity=-1,
            )

        auc, pr, ll = cross_val_metrics(lgbm_fn, X, y)
        print(f"  CV → AUC: {auc:.5f} | PR-AUC: {pr:.5f} | LogLoss: {ll:.5f}")
        mlflow.log_metrics({"lgbm_cv_auc": auc, "lgbm_cv_pr_auc": pr, "lgbm_cv_logloss": ll})

        lgbm = lgbm_fn()
        lgbm.fit(X, y)

        # ── XGBoost ───────────────────────────────────────────────────────────
        print("\n── XGBoost (tuning) ──")
        xgb_params = tune_xgb(X, y)
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})

        def xgb_fn():
            return XGBClassifier(
                **xgb_params, scale_pos_weight=pos_weight,
                tree_method="hist", random_state=RANDOM_STATE, verbosity=0,
            )

        auc, pr, ll = cross_val_metrics(xgb_fn, X, y)
        print(f"  CV → AUC: {auc:.5f} | PR-AUC: {pr:.5f} | LogLoss: {ll:.5f}")
        mlflow.log_metrics({"xgb_cv_auc": auc, "xgb_cv_pr_auc": pr, "xgb_cv_logloss": ll})

        xgb = xgb_fn()
        xgb.fit(X, y)

        # ── CatBoost ──────────────────────────────────────────────────────────
        print("\n── CatBoost ──")
        cat = train_catboost(X, y)
        df_cb  = to_catboost_df(X)

        # CV manual para CatBoost (necesita DataFrame)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cat_losses = []
        for tr_idx, val_idx in skf.split(X, y):
            m = CatBoostClassifier(
                iterations=1000, learning_rate=0.05, depth=8,
                cat_features=CATEG_COLS, auto_class_weights="Balanced",
                random_seed=RANDOM_STATE, verbose=False,
            )
            m.fit(df_cb.iloc[tr_idx], y[tr_idx])
            p = m.predict_proba(df_cb.iloc[val_idx])[:, 1]
            cat_losses.append(log_loss(y[val_idx], p))
        cat_ll  = np.mean(cat_losses)
        cat_auc = roc_auc_score(y, cat.predict_proba(df_cb)[:, 1])
        mlflow.log_params({
            "catboost_iterations": 1000,
            "catboost_lr": 0.05,
            "catboost_depth": 8,
        })
        mlflow.log_metrics({"catboost_cv_logloss": cat_ll, "catboost_train_auc": cat_auc})
        print(f"  CV LogLoss: {cat_ll:.5f} | Train AUC: {cat_auc:.5f}")

        # ── Ensemble CV ───────────────────────────────────────────────────────
        print("\n── Validando ensemble (blend3) ──")
        # Estimar ensemble loss sumando las predicciones CV
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        tr_idx, val_idx = next(sss.split(X, y))

        lgbm_tmp = lgbm_fn(); lgbm_tmp.fit(X[tr_idx], y[tr_idx])
        xgb_tmp  = xgb_fn();  xgb_tmp.fit(X[tr_idx],  y[tr_idx])
        cat_df   = to_catboost_df(X)
        cat_tmp  = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=8,
            cat_features=CATEG_COLS, auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=False,
        )
        cat_tmp.fit(cat_df.iloc[tr_idx], y[tr_idx])

        val_df   = to_catboost_df(X[val_idx])
        p_blend  = (
            lgbm_tmp.predict_proba(X[val_idx])[:, 1]
            + xgb_tmp.predict_proba(X[val_idx])[:, 1]
            + cat_tmp.predict_proba(val_df)[:, 1]
        ) / 3

        ensemble_auc = roc_auc_score(y[val_idx], p_blend)
        ensemble_pr  = average_precision_score(y[val_idx], p_blend)
        ensemble_ll  = log_loss(y[val_idx], p_blend)

        print(f"  Ensemble → AUC: {ensemble_auc:.5f} | PR-AUC: {ensemble_pr:.5f} | LogLoss: {ensemble_ll:.5f}")
        mlflow.log_metrics({
            "ensemble_val_auc":     ensemble_auc,
            "ensemble_val_pr_auc":  ensemble_pr,
            "ensemble_val_logloss": ensemble_ll,
        })

        # ── Quality gate ──────────────────────────────────────────────────────
        passed_gate = ensemble_ll <= MAX_LOG_LOSS
        mlflow.log_metric("quality_gate_passed", int(passed_gate))
        print(f"\n  Quality gate (LogLoss ≤ {MAX_LOG_LOSS}): {'✓ PASS' if passed_gate else '✗ FAIL'}")

        # ── Loggear ensemble como PyfuncModel ─────────────────────────────────
        print("\n── Registrando ensemble en MLflow ──")
        ensemble = ChurnEnsemble(lgbm, xgb, cat, FEATURE_COLS_V5, CATEG_COLS)

        signature = mlflow.models.infer_signature(
            pd.DataFrame(X[:5], columns=FEATURE_COLS_V5),
            ensemble.predict(None, pd.DataFrame(X[:5], columns=FEATURE_COLS_V5)),
        )

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ensemble,
            signature=signature,
            registered_model_name=MODEL_NAME,
            pip_requirements=[
                "lightgbm>=4.0", "xgboost>=2.0", "catboost>=1.2",
                "scikit-learn", "pandas", "numpy",
            ],
        )

        # ── Loggear artefactos adicionales ────────────────────────────────────
        mlflow.log_dict(
            {"feature_cols": FEATURE_COLS_V5, "categ_cols": CATEG_COLS},
            "feature_schema.json",
        )

        run_id = run.info.run_id
        print(f"\n  Run ID:  {run_id}")
        print(f"  Modelo:  {MODEL_NAME}")

    # ── Transición a Staging si pasa el quality gate ──────────────────────────
    if passed_gate:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest   = max(versions, key=lambda v: int(v.version))

        client.transition_model_version_stage(
            name=MODEL_NAME, version=latest.version, stage="Staging",
        )
        print(f"\n✓ Modelo {MODEL_NAME} v{latest.version} promovido a Staging")
    else:
        print(f"\n✗ Quality gate fallido — modelo NO promovido (LogLoss={ensemble_ll:.4f} > {MAX_LOG_LOSS})")

    print(f"\n  Ver en: {cfg.mlflow_tracking_uri}/#/experiments")


if __name__ == "__main__":
    main()
