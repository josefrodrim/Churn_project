# Datos crudos — KKBox Churn Prediction Challenge

Los archivos con sufijo `_v2` corresponden a la segunda fase del concurso
(extensión de un mes). Los nombres originales de Kaggle se mantienen intactos
porque están trackeados por DVC.

## Mapa temporal

```
Ene 2015 ──────────────────────────────────────── Mar 2017

│◄──────────────── transactions.csv ────────────►│ (hasta Feb 28)
│◄──────────────── transactions_v2.csv ──────────────►│ (hasta Mar 31)

│◄──────────────── user_logs.csv ────────────────►│ (hasta Feb 28, ~392M filas)
                                                   │◄── user_logs_v2.csv ──►│ (Mar, 18M filas)

│◄──────────────── members_v3.csv ───────────────────────────────►│ (registro hasta Abr 2017)
```

## Archivos de etiquetas

| Archivo | Registros | Período que predice | Churn rate |
|---|---|---|---|
| `train.csv` | 992,931 | Febrero 2017 | 6.39% |
| `train_v2.csv` | 970,960 | Marzo 2017 | 8.99% |
| `sample_submission_v2.csv` | 907,471 | Abril 2017 | — (sin etiqueta) |

> **Confusión común:** `train_v2.csv` tiene etiquetas, por eso Kaggle lo llama
> "train". Pero temporalmente es el mes siguiente a `train.csv`, por lo que
> funciona mejor como **validación temporal** si ya entrenas en `train.csv`.

## Split temporal recomendado

```
ENTRENAR   →  train.csv       (feb 2017)  +  logs/tx hasta Feb 28
VALIDAR    →  train_v2.csv    (mar 2017)  +  logs/tx hasta Mar 31
PREDECIR   →  submission_v2   (abr 2017)  +  logs/tx hasta Mar 31
```

Para el modelo final (submission), se puede reentrenar en **feb + mar** juntos
antes de predecir abril — más datos, mejor generalización.

## Archivos de comportamiento

| Archivo | Registros | Ventana | Descripción |
|---|---|---|---|
| `transactions.csv` | 21,547,746 | Ene 2015 – Feb 2017 | Historial de pagos y renovaciones |
| `transactions_v2.csv` | 1,431,009 | Ene 2015 – Mar 2017 | Extensión de transacciones |
| `user_logs.csv` | 392,106,543 | Ene 2015 – Feb 2017 | Comportamiento de escucha diario (~28 GB) |
| `user_logs_v2.csv` | 18,396,362 | Mar 2017 | Extensión de logs (solo marzo) |
| `members_v3.csv` | 6,769,473 | Registro: 2004–2017 | Datos demográficos de usuarios |

## Datos procesados (generados, no subir a git)

| Archivo | Descripción |
|---|---|
| `processed/features_train.parquet` | Features para train.csv (feb) |
| `processed/features_test.parquet` | Features para train_v2.csv (mar) |
| `processed/user_logs_agg.parquet` | Agregado de user_logs (hasta Feb 28) |
| `processed/user_logs_agg_mar.parquet` | Agregado con logs de marzo |
| `processed/user_logs_agg_mar_v2.parquet` | Agregado con ventanas 7d/30d/90d |
