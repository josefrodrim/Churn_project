# Hallazgos del Proyecto — KKBox Churn Prediction

## Dataset
- **Fuente:** Kaggle — WSDM KKBox's Churn Prediction Challenge
- **Descargado:** 2026-04-24
- **Tamaño ZIP original:** 8.3GB

---

## Archivos y tamaños descomprimidos

| Archivo | Tamaño | Filas | Notas |
|---|---|---|---|
| `train.csv` | 45MB | 992,931 | Target Feb 2017 |
| `train_v2.csv` | 44MB | 970,960 | Target Mar 2017 |
| `members_v3.csv` | 408MB | 6,769,473 | Sin columna expiration_date |
| `transactions.csv` | 1.6GB | ~21.5M | Hasta 2017-02-28 |
| `transactions_v2.csv` | 110MB | ~1.5M | Hasta 2017-03-31 |
| `user_logs.csv` | 28GB | ~400M | Hasta 2017-02-28 |
| `user_logs_v2.csv` | 1.3GB | ~30M | Hasta 2017-03-31 |
| `sample_submission_v2.csv` | 41MB | ~1M | Test set Abr 2017 |

---

## Definición del problema

- **Target:** `is_churn = 1` si el usuario NO renovó dentro de 30 días tras la expiración
- **Train set:** usuarios cuya membresía expira en **febrero 2017**
- **Test set:** usuarios cuya membresía expira en **marzo 2017**
- `is_cancel = 1` **NO** implica churn — un usuario puede cancelar y re-suscribirse dentro de 30 días

---

## Hallazgos EDA — Overview General

### Target Variable (train / train_v2)

| Métrica | Train (Feb 2017) | Train v2 (Mar 2017) |
|---|---|---|
| Usuarios | 992,931 | 970,960 |
| Churn rate | **6.39%** | **8.99%** |
| Usuarios en común | 881,701 | — |

- **Desbalance severo:** ~93% renewal vs ~7% churn — requiere estrategia de balanceo (imbalanced-learn)
- El churn rate subió de 6.4% a 9% entre febrero y marzo 2017
- 881,701 usuarios se repiten entre ambos sets — hay solapamiento significativo

---

### Members (members_v3.csv)

| Métrica | Valor |
|---|---|
| Total usuarios | 6,769,473 |
| Nulos en `gender` | **4,429,505 (65.4%)** |
| Edad válida (0 < bd < 100) | **32.85%** de filas |

- `gender`: 65.4% de valores son nulos — no es una columna confiable para modelar directamente
- `bd` (edad): solo el 32.85% tiene valores válidos, el resto son outliers o ceros
  - Distribución de género válido: male 1,195,355 / female 1,144,613 (casi igual)
- Sin nulos en ciudad, método de registro ni fecha de registro

**Acciones requeridas:**
- Tratar `bd`: filtrar outliers (fuera del rango 1–100) e imputar o crear flag `age_valid`
- Tratar `gender`: crear categoría `unknown` para los nulos

---

### Transactions (transactions.csv + v2)

| Métrica | Valor |
|---|---|
| Total transacciones | 22,978,755 |
| Usuarios únicos | 2,426,143 |
| Auto-renew rate | **84.78%** |
| Cancel rate | **3.88%** |
| Nulos | 0 en todas las columnas |

- La gran mayoría de usuarios tiene auto-renew activo (84.78%)
- Solo el 3.88% de transacciones son cancelaciones activas
- Sin nulos — tabla limpia para modelar

---

### User Logs (muestra 500K filas)

| Métrica | Valor |
|---|---|
| Usuarios únicos (muestra) | 100,408 |
| Rango de fechas | 2015-01-01 → 2017-02-28 |
| Nulos | 0 en todas las columnas |

- Cubre más de 2 años de comportamiento de escucha
- Sin nulos — tabla limpia
- Por tamaño (28GB) requiere procesamiento por chunks o agregación previa al join

---

### Cobertura de Keys entre tablas

| Relación | Cobertura |
|---|---|
| Usuarios train en `members` | **88.34%** |
| Usuarios train en `transactions` | **100%** |

- Todos los usuarios del train tienen historial de transacciones ✓
- 11.66% de usuarios del train **no tienen** registro en members — hay que decidir cómo manejar estos casos (imputar o crear features sin demografía)

---

## Decisiones tomadas

| Fecha | Decisión | Razón |
|---|---|---|
| 2026-04-24 | Usar `members_v3.csv` | Es el refresh más reciente, sin expiration_date |
| 2026-04-24 | Concatenar `transactions.csv` + `_v2` | Cubrir hasta mar 2017 (23M filas total) |
| 2026-04-24 | Concatenar `user_logs.csv` + `_v2` | Cubrir período completo |
| 2026-04-24 | Samplear user_logs para EDA (500K filas) | Archivo de 28GB, no cargable completo en RAM |

---

## Problemas de calidad de datos identificados

| Tabla | Columna | Problema | Acción sugerida |
|---|---|---|---|
| `members_v3` | `gender` | 65.4% nulos | Crear categoría `unknown` |
| `members_v3` | `bd` (edad) | Solo 32.85% válidos (0 < bd < 100) | Filtrar outliers, flag `age_valid` |
| `train` / `train_v2` | `is_churn` | Desbalance 93/7 | SMOTE, class_weight, o undersampling |
| `members_v3` | — | 11.66% de usuarios train sin registro | Imputar o crear features sin demografía |

---

---

## Hallazgos EDA — Transacciones (transactions.csv + v2)

### Overview

| Métrica | Valor |
|---|---|
| Total transacciones | 22,978,755 |
| Usuarios únicos | 2,426,143 |
| Rango de fechas | 2015-01-01 → 2017-03-31 |
| Plan más común | 30 días (20.2M transacciones, 87.8%) |
| Precio=0 (free/promo) | 6.6% de transacciones |
| Descuento medio | 2.9% sobre plan_list_price |

### Distribución de planes

| Días | Transacciones | Notas |
|---|---|---|
| 0 | 872,342 | Cancelaciones inmediatas |
| 7 | 589,807 | Plan semanal |
| 30 | 20,174,288 | Plan mensual (dominante) |
| 180 | 76,172 | Plan semestral |
| 195 | 138,802 | |
| 410 | 162,236 | Plan anual+ |

### Churn vs Renewal — features de transacciones

| Feature | Renewal (0) | Churn (1) | Diferencia |
|---|---|---|---|
| n_transactions | 17.5 | 11.4 | **-34.7%** |
| n_cancels | 0.29 | 0.62 | **+117%** |
| ever_canceled | 23.2% | 48.9% | **+111%** |
| avg_discount_pct | 0.7% | 1.7% | +148% |
| avg_plan_days | 31.7 | 62.7 | +98% |
| avg_price (NTD) | 138.9 | 272.7 | +96% |
| **last_is_cancel** | **2.2%** | **39.3%** | **+1,697%** ← señal más fuerte |
| **last_is_auto_renew** | **91.9%** | **42.3%** | **-54%** ← segunda señal más fuerte |
| last_plan_days | 32.9 | 71.1 | +116% |
| last_price (NTD) | 140.4 | 304.7 | +117% |
| price_trend (NTD) | -2.4 | +40.5 | +1,768% |

### Interpretaciones clave

- **`last_is_cancel`**: el 39% de churners tienen su ÚLTIMA transacción como cancelación — señal más fuerte de todas
- **`last_is_auto_renew`**: solo el 42% de churners tenían auto-renew activo en su última tx (vs 92% de renewals)
- Churners tienden a estar en **planes más largos y caros** (anuales o multi-mes) — probable correlación con planes corporativos o promociones
- Churners muestran **tendencia de precio positiva**: subieron de plan antes de irse (+40 NTD de tendencia)
- Churners tienen **menos transacciones totales** (-34.7%) — son usuarios menos recurrentes o más recientes
- Churners reciben **más descuentos** (+148%) — pueden ser usuarios captados por promociones (price-sensitive)

### Figuras generadas

- `reports/figures/tx_plan_days.png`
- `reports/figures/tx_price_distribution.png`
- `reports/figures/tx_churn_vs_renewal.png`
- `reports/figures/tx_cancel_autorenew_churn.png`
- `reports/figures/tx_price_trend_churn.png`
- `reports/figures/tx_churn_by_plan_days.png`
- `reports/figures/tx_churn_by_payment_method.png`

---

## Hallazgos EDA — Members (members_v3.csv cruzado con train)

### Overview

| Métrica | Valor |
|---|---|
| Usuarios en análisis (train ∩ members) | 877,161 |
| Churn rate global | **6.57%** |

### Churn rate por método de registro

| registered_via | Churn rate | n |
|---|---|---|
| 4 | **18.3%** | 49,283 |
| 3 | **12.7%** | 105,445 |
| 9 | 8.6% | 236,620 |
| 13 | 8.5% | 3,087 |
| 7 | **3.0%** | 482,726 |

- Los métodos 4 y 3 tienen churn 2–3× el promedio; el método 7 (el más común, 55% de usuarios) tiene churn muy bajo

### Churn rate por género

| Género | Churn rate | n |
|---|---|---|
| male | 8.8% | 206,284 |
| female | 8.6% | 185,408 |
| unknown | **4.8%** | 485,469 |

- Los usuarios sin género declarado (55% del set de análisis) tienen churn notablemente más bajo — posiblemente usuarios más "pasivos" o con cuentas corporativas

### Churn rate por grupo de edad

| Edad | Churn rate | n |
|---|---|---|
| <18 | **19.3%** | 17,216 |
| 18-25 | **11.5%** | 117,439 |
| 25-35 | 7.1% | 169,568 |
| 35-45 | 6.8% | 59,095 |
| 45-55 | 5.8% | 20,373 |
| 55+ | 6.1% | 5,443 |

- Churn disminuye monotónicamente con la edad (excepto 55+): los más jóvenes se van más

### Churn rate por antigüedad

| Antigüedad | Churn rate | n |
|---|---|---|
| <3m | 9.5% | 33,506 |
| 3-12m | 6.0% | 141,963 |
| 1-2y | 6.8% | 197,057 |
| 2-3y | **8.4%** | 85,046 |
| 3-5y | 6.9% | 194,813 |
| 5y+ | 5.3% | 224,755 |

- El churn es más alto en recién llegados (<3m) y hay un pico en 2-3 años — posiblemente contratos que expiran
- Los usuarios con 5y+ de antigüedad son los más fieles (5.3%)

### Figuras generadas

- `reports/figures/mem_churn_by_city.png`
- `reports/figures/mem_churn_by_registration.png`
- `reports/figures/mem_churn_demographics.png`
- `reports/figures/mem_tenure_city.png`

---

## Hallazgos EDA — User Logs (user_logs.csv — 392M filas, 5.2M usuarios)

### Notas de procesamiento

- `user_logs_v2.csv` cubre íntegramente marzo 2017 (post-cutoff 2017-02-28) → se descarta para evitar leakage
- Outliers en `total_secs`: valores ±9.22×10¹⁵ (sentinels de int64) → clippeados a [0, 86400]
- Usuarios en análisis (train ∩ user_logs): **869,926** | churn rate: 6.57%

### Churn vs Renewal — features de escucha

| Feature | Renewal (0) | Churn (1) | Diferencia |
|---|---|---|---|
| n_days (días activos) | 285.3 | 239.6 | **-16.0%** |
| avg_daily_secs | 6,529 | 6,755 | +3.5% ← leve |
| avg_daily_completed (canciones) | 24.5 | 25.2 | +2.8% ← leve |
| avg_daily_unq (canciones únicas) | 25.0 | 26.3 | +5.1% ← leve |
| completion_ratio | 68.0% | 67.5% | -0.7% ← irrelevante |
| **days_since_last** | **24.7** | **40.3** | **+63.6% ← señal fuerte** |
| **listening_trend** | **-772** | **-2,064** | **-167% ← señal fuerte** |

### Churn rate por recencia del último log

| Recencia | Churn rate | n |
|---|---|---|
| <1 semana | **4.6%** | 255,776 |
| 1 semana – 1 mes | **23.7%** | 91,081 |
| 1 – 2 meses | **19.5%** | 28,089 |
| 2 – 3 meses | 8.9% | 12,167 |
| 3 – 6 meses | 8.1% | 19,114 |
| 6 meses+ | 8.7% | 41,413 |

### Interpretaciones clave

- **`days_since_last`** (+63.6%): churners tuvieron su último log hace ~40 días vs ~25 de los renewals — señal de desenganche
- **`listening_trend`** (-167%): ambos grupos muestran caída en los últimos 30 días (posiblemente estacional), pero churners caen 2.7× más — señal de abandono inminente
- **Comportamiento "last hurrah"**: el pico de churn en 1w–1m (23.7%) vs <1w (4.6%) indica que hay usuarios que usan el servicio por última vez ~2 semanas antes de que expire su membresía
- `avg_daily_secs` levemente mayor en churners (+3.5%): cuando usan el servicio, lo usan intensamente — el problema es que dejan de usarlo
- `n_days` menor en churners (-16%): menor constancia histórica

### Figuras generadas

- `reports/figures/ul_churn_vs_renewal.png`
- `reports/figures/ul_listening_trend.png`
- `reports/figures/ul_churn_by_recency.png`

---

## Hallazgos — Modelado (src/models/train_06.py)

### Configuración

- Split: 80% train (794,344) / 20% test (198,587), estratificado
- CV: 5-fold StratifiedKFold
- Desbalance: `class_weight='balanced'` (LR), `scale_pos_weight=14.6` (XGBoost), `is_unbalance=True` (LightGBM)

### Resultados CV (5-fold sobre train)

| Modelo | ROC-AUC | ± | PR-AUC | ± |
|---|---|---|---|---|
| **LightGBM** | **0.9857** | 0.0002 | **0.8559** | 0.0025 |
| XGBoost | 0.9856 | 0.0002 | 0.8526 | 0.0026 |
| Logistic Regression | 0.9650 | 0.0003 | 0.5773 | 0.0035 |

### Evaluación en test set

| Modelo | ROC-AUC | PR-AUC | F1 (umbral óptimo) |
|---|---|---|---|
| **LightGBM** | **0.9853** | **0.8549** | **0.760** (thr=0.89) |
| XGBoost | 0.9850 | 0.8505 | 0.756 (thr=0.90) |
| Logistic Regression | 0.9638 | 0.5710 | 0.611 (thr=0.88) |

### Classification report — LightGBM (test)

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| Renewal | 0.982 | 0.986 | 0.984 |
| Churn | 0.779 | 0.741 | 0.760 |

### Interpretaciones clave

- LightGBM y XGBoost prácticamente idénticos (diferencia de 0.0003 AUC) — ambos dominan sobre LR
- La LR tiene PR-AUC muy bajo (0.57 vs 0.85) — no captura bien la clase minoritaria a pesar del balanceo
- Precision churn = 0.779: de cada 10 usuarios predichos como churners, ~8 realmente churnan
- Recall churn = 0.741: detectamos el 74% de los churners reales
- **Modelo guardado:** `models/best_model.joblib` (LightGBM)

### Figuras generadas

- `reports/figures/model_roc_pr.png`
- `reports/figures/model_shap.png`

---

## Hallazgos — Análisis de Errores (src/models/error_analysis_08.py)

### Matriz de confusión — LightGBM (test set, umbral=0.891)

| | Predicho: Renewal | Predicho: Churn |
|---|---|---|
| **Real: Renewal** | TN = 183,227 | FP = 2,666 |
| **Real: Churn** | FN = 3,282 | TP = 9,412 |

### Perfil de Falsos Negativos (FN = 3,282 churners no detectados)

| Feature | FN | TP (detectados) | Diferencia |
|---|---|---|---|
| last_is_cancel | 0.126 | 0.472 | FN casi no cancelaron |
| last_is_auto_renew | 0.214 | 0.485 | FN tienen menos auto-renew activo |
| days_since_last | 5.0 | 46.9 | **FN estuvieron activos recién** |
| n_transactions | 12.2 | 11.1 | Similar |

**Conclusión FN:** Son churners "silenciosos" — estuvieron activos en los logs hasta el final (days_since_last=5) pero aun así no renovaron. El modelo no los detecta porque su señal de comportamiento es ambigua: ni cancelo, ni tiene auto-renew claro, ni está inactivo.

### Perfil de Falsos Positivos (FP = 2,666 renewals mal clasificados)

| Feature | FP | Interpretación |
|---|---|---|
| proba_churn media | 0.940 | Muy alta confianza en la predicción errónea |
| last_is_cancel | 0.458 | Cancelaron su última tx (como churners reales) |
| last_is_auto_renew | 0.466 | Sin auto-renew activo |
| days_since_last | 26.1 | Inactivos por un mes |

**Conclusión FP:** Son renewals que se comportan idénticamente a churners (cancelaron, sin auto-renew, inactivos) pero al final renovaron. Difíciles de separar sin información adicional (e.g., historial de reactivaciones).

### Figuras generadas

- `reports/figures/error_confusion_matrix.png`
- `reports/figures/error_distributions.png`
- `reports/figures/error_scores.png`

---

## Próximos pasos

- [x] EDA profundo de transacciones ✓
- [x] EDA demografía de members ✓
- [x] EDA profundo de user_logs ✓
- [x] Feature engineering: 992K usuarios × 30 features ✓
- [x] Modelado: LightGBM ROC-AUC 0.9853, F1 churn 0.760 ✓
- [x] Análisis de errores: FN son churners silenciosos (activos hasta el final) ✓
---

## Hallazgos — Test Set Features (src/features/build_test_features_09.py)

### Dataset: train_v2.csv (predicción marzo 2017)

| Métrica | Valor |
|---|---|
| Usuarios | 970,960 |
| Churn rate | **8.99%** (vs 6.39% en train) |
| Fecha referencia members/tenure | 2017-03-31 |
| Caché user_logs reutilizado | ✓ |
| Nulos tras imputación | 0 |
| Guardado en | `data/processed/features_test.parquet` |

- El churn rate sube de 6.4% a 9% — marzo 2017 tiene más churners que febrero
- Mismos 30 features que el train set, con `tenure_days` calculado a 2017-03-31

---

## Próximos pasos

- [x] EDA profundo de transacciones ✓
- [x] EDA demografía de members ✓
- [x] EDA profundo de user_logs ✓
- [x] Feature engineering: 992K usuarios × 30 features ✓
- [x] Modelado: LightGBM ROC-AUC 0.9853, F1 churn 0.760 ✓
- [x] Análisis de errores: FN son churners silenciosos (activos hasta el final) ✓
- [x] Features para test set (train_v2): 970K usuarios × 30 features ✓
- [ ] Hyperparameter tuning (Optuna — en curso)
