# Submissions — KKBox Churn Prediction Challenge

Todas las submissions son para `sample_submission_v2.csv` (907,471 usuarios,
predicción de churn en **abril 2017**). Métrica: **log loss** (menor es mejor).

## Historial

| Archivo | Public Score | Descripción |
| --- | --- | --- |
| `submission_lgbm_tuned.csv` | 0.37856 | LightGBM tuneado (Optuna), entrenado en feb, 30 features base |
| `submission_combined_v2.csv` | 0.61068 | ❌ Intento de combinar feb+mar con offset incorrecto en `days_since_last` |
| `submission_v3_fixed.csv` | 0.30398 | Fix del offset: referencia fija `LOG_CUTOFF=Mar31` para `days_since_last` |
| `submission_v4_expiry.csv` | 0.23528 | +6 features: `days_until_expire`, `is_expired`, `prev_churn` y otros |
| `submission_v5_full.csv` | 0.29875 | ❌ Calibración isotónica sobreajustó el holdout |
| `submission_v5_raw.csv` | **0.23504** | v5 sin calibración — 48 features, mejor score actual |
| `submission_v6_temporal.csv` | 0.24426 | Split temporal correcto: entrenado en feb+mar por separado — 36 features |

| `submission_v7_xgb.csv` | 0.27209 | XGBoost solo — tuneado en 25% subsample, underpredicts (mean 0.049) |
| `submission_v8_catboost.csv` | 0.23963 | CatBoost solo — ordered boosting, cats nativas |
| `submission_v9_blend_lgbm_xgb.csv` | 0.23945 | Blend LightGBM + XGBoost — XGB arrastra el score |
| `submission_v10_blend3.csv` | **0.23412** | Blend LightGBM + XGBoost + CatBoost — nuevo mejor |

## Mejor submission

`submission_v10_blend3.csv` — Public 0.23412 / Private 0.23436

## Lecciones

- `days_until_expire` es el feature más importante (membresía vencida = churn casi seguro)
- Los offsets de fecha entre training y submission arruinan la calibración del modelo
- La calibración isotónica sobreajusta cuando el holdout es aleatorio (no temporal)
- El split temporal correcto es: entrenar feb → validar mar → predecir abr
- El blend de 3 modelos (LGBM + XGB + CatBoost) mejora marginalmente sobre el mejor modelo solo
- XGBoost tuneado en subsample pequeño (25%) subpredice — necesita tuning en dataset completo
- CatBoost solo no supera a LGBM, pero añade diversidad útil en el blend
