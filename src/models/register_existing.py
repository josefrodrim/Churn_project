"""
Registra los modelos entrenados existentes en MLflow Model Registry
sin necesidad de reentrenar desde los datos crudos.

Carga lgbm_v5 + xgb_v1 + catboost_v1, los empaqueta como ChurnEnsemble
(mismo PyfuncModel que train_mlflow_17) y los registra bajo 'churn-ensemble'.
Las métricas provienen del set de test temporal (Mar 2017, ~198K usuarios).

Ejecutar:
    python -m src.models.register_existing
"""

import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.api.config import get_settings
from src.models.train_mlflow_17 import ChurnEnsemble
from src.models.ensemble_16 import CATEG_COLS
from src.models.retrain_submit_14 import FEATURE_COLS_V5

MODELS_DIR     = ROOT / "models"
EXPERIMENT_NAME = "churn-kkbox"
MODEL_NAME      = "churn-ensemble"

# Métricas reales evaluadas en test set temporal (Mar 2017, ~198K usuarios).
# ROC-AUC y PR-AUC: notebook 07 / notebook 08.
# LogLoss: score privado Kaggle del blend 3 modelos (v10).
KNOWN_METRICS = {
    "test_roc_auc":  0.9853,
    "test_pr_auc":   0.8549,
    "test_log_loss": 0.23412,
    "test_f1":       0.7599,
    "n_test_rows":   198587,
    "churn_rate":    0.0639,
}

BLEND_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]


def load_models():
    lgbm_artifact = joblib.load(MODELS_DIR / "lgbm_v5.joblib")
    xgb_artifact  = joblib.load(MODELS_DIR / "xgb_v1.joblib")

    cb = CatBoostClassifier()
    cb.load_model(str(MODELS_DIR / "catboost_v1.cbm"))

    assert lgbm_artifact["features"] == FEATURE_COLS_V5, "LGBM features mismatch"
    assert xgb_artifact["features"]  == FEATURE_COLS_V5, "XGB features mismatch"
    assert list(cb.feature_names_)   == FEATURE_COLS_V5, "CatBoost features mismatch"

    return lgbm_artifact["model"], xgb_artifact["model"], cb


def main():
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("═" * 60)
    print("REGISTRO DE MODELOS EXISTENTES → MLflow Registry")
    print(f"Tracking URI: {cfg.mlflow_tracking_uri}")
    print("═" * 60)

    print("\n── Cargando modelos ──")
    lgbm, xgb, catboost = load_models()
    print(f"  LGBM  : {type(lgbm).__name__}")
    print(f"  XGB   : {type(xgb).__name__}")
    print(f"  CatBoost: {type(catboost).__name__}")

    ensemble = ChurnEnsemble(
        lgbm=lgbm,
        xgb=xgb,
        catboost=catboost,
        feature_cols=FEATURE_COLS_V5,
        categ_cols=CATEG_COLS,
        blend_weights=BLEND_WEIGHTS,
    )

    print("\n── Iniciando run de MLflow ──")
    with mlflow.start_run(run_name="ensemble-register-existing") as run:
        print(f"  run_id: {run.info.run_id[:8]}...")

        mlflow.log_params({
            "n_features":    len(FEATURE_COLS_V5),
            "blend_weights": "1/3-1/3-1/3",
            "lgbm_source":   "lgbm_v5.joblib",
            "xgb_source":    "xgb_v1.joblib",
            "catboost_source": "catboost_v1.cbm",
            "eval_period":   "Mar-2017",
        })
        mlflow.log_metrics(KNOWN_METRICS)

        print("\n── Logging modelo como pyfunc ──")
        # MLflow 3.x log_model internamente usa "logged_model" que resuelve el
        # artifact root como path local. Usamos la API tradicional de runs:
        # save_model en /tmp → log_artifacts via HTTP → register_model.
        run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "ensemble"
            mlflow.pyfunc.save_model(
                path=str(model_path),
                python_model=ensemble,
                input_example=pd.DataFrame(
                    [np.zeros(len(FEATURE_COLS_V5))],
                    columns=FEATURE_COLS_V5,
                ),
            )
            mlflow.log_artifacts(str(model_path), artifact_path="ensemble")

        model_uri = f"runs:/{run_id}/ensemble"
        print(f"  artifact URI: {model_uri}")

    # Registrar en el Model Registry y aplicar quality gate
    log_loss = KNOWN_METRICS["test_log_loss"]
    max_ll   = cfg.max_log_loss

    print(f"\n── Registrando en Model Registry ──")
    client = mlflow.MlflowClient()
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"  Versión registrada: {mv.version}")

    print(f"\n── Quality gate: LogLoss {log_loss:.5f} ≤ {max_ll} ──")
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = sorted(versions, key=lambda v: int(v.version))[-1]

    if log_loss <= max_ll:
        # transition_model_version_stage es la API que usa dependencies.py (stage-based).
        # En MLflow 3.x está deprecated pero sigue funcionando para compatibilidad.
        client.transition_model_version_stage(MODEL_NAME, latest.version, stage="Production")
        print(f"  ✓ PASA — versión {latest.version} promovida a Production")
    else:
        print(f"  ✗ NO PASA — versión {latest.version} queda en None")

    print(f"\n  Modelo: {MODEL_NAME} v{latest.version}")
    print(f"  ROC-AUC  : {KNOWN_METRICS['test_roc_auc']:.4f}")
    print(f"  PR-AUC   : {KNOWN_METRICS['test_pr_auc']:.4f}")
    print(f"  LogLoss  : {KNOWN_METRICS['test_log_loss']:.5f}")
    print("═" * 60)


if __name__ == "__main__":
    main()
