-- Schema principal del sistema de churn prediction
-- Ejecutado automáticamente por el contenedor postgres al inicializarse

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── FEATURES STORE ────────────────────────────────────────────────────────────
-- Almacena las 48 features v5 pre-computadas por usuario y período mensual.
-- El pipeline batch escribe aquí; la API lee desde aquí para predicción rápida.

CREATE TABLE IF NOT EXISTS features_monthly (
    id              BIGSERIAL PRIMARY KEY,
    msno            TEXT        NOT NULL,
    period          TEXT        NOT NULL,  -- 'YYYY-MM', e.g. '2017-03'
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- transacciones base
    n_transactions      REAL, n_cancels          REAL, ever_canceled       SMALLINT,
    avg_discount_pct    REAL, avg_plan_days       REAL, avg_price           REAL,
    n_unique_plans      REAL, n_payment_methods   REAL,
    last_is_cancel      SMALLINT, last_is_auto_renew SMALLINT,
    last_plan_days      REAL, last_price          REAL, last_list_price     REAL,
    price_trend         REAL, last_payment_method SMALLINT,

    -- members
    city              SMALLINT, registered_via    SMALLINT, gender_enc       SMALLINT,
    age               REAL,     bd_valid           SMALLINT, tenure_days      REAL,
    has_member_record SMALLINT,

    -- user logs base
    n_days              REAL, avg_daily_secs       REAL, avg_daily_completed REAL,
    avg_daily_unq       REAL, completion_ratio      REAL,
    days_since_last     REAL, listening_trend       REAL, has_log_record     SMALLINT,

    -- expiry + churn lag
    days_until_expire    REAL,    is_expired            SMALLINT,
    auto_renew_at_expire SMALLINT, cancel_at_expire     SMALLINT,
    n_renewals           REAL,    prev_churn            SMALLINT,

    -- tx recency
    days_since_last_tx REAL, had_tx_last_7d SMALLINT,
    had_tx_last_30d    SMALLINT, n_tx_last_30d REAL,

    -- cancel before expire
    cancel_before_expire SMALLINT,

    -- multi-window logs
    n_days_7d REAL, secs_per_day_7d REAL,
    n_days_90d REAL, secs_per_day_90d REAL,
    trend_7d REAL, trend_7d_vs_30d REAL,

    UNIQUE (msno, period)
);

CREATE INDEX IF NOT EXISTS idx_features_period ON features_monthly (period);
CREATE INDEX IF NOT EXISTS idx_features_msno   ON features_monthly (msno);


-- ── PREDICTIONS ───────────────────────────────────────────────────────────────
-- Registro de cada predicción generada (batch o API en tiempo real).
-- Es la fuente de verdad para monitoreo de output drift y auditoría.

CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    msno          TEXT        NOT NULL,
    churn_prob    REAL        NOT NULL CHECK (churn_prob BETWEEN 0 AND 1),
    churn_label   SMALLINT    NOT NULL CHECK (churn_label IN (0, 1)),
    model_version TEXT        NOT NULL,
    period        TEXT        NOT NULL,  -- período predicho, e.g. '2017-04'
    source        TEXT        NOT NULL DEFAULT 'api' CHECK (source IN ('api', 'batch')),
    predicted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_period  ON predictions (period);
CREATE INDEX IF NOT EXISTS idx_predictions_msno    ON predictions (msno);
CREATE INDEX IF NOT EXISTS idx_predictions_version ON predictions (model_version);


-- ── MODEL VERSIONS ────────────────────────────────────────────────────────────
-- Registro de qué modelo está activo en cada stage.
-- MLflow es la fuente de verdad; esta tabla es una proyección para consultas rápidas.

CREATE TABLE IF NOT EXISTS model_versions (
    id             BIGSERIAL PRIMARY KEY,
    name           TEXT        NOT NULL,
    version        TEXT        NOT NULL,
    stage          TEXT        NOT NULL CHECK (stage IN ('None', 'Staging', 'Production', 'Archived')),
    log_loss_val   REAL,
    roc_auc_val    REAL,
    feature_count  SMALLINT,
    registered_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    promoted_at    TIMESTAMPTZ,
    promoted_by    TEXT,

    UNIQUE (name, version)
);


-- ── GROUND TRUTH ──────────────────────────────────────────────────────────────
-- Cuando llegan las etiquetas reales (1 mes después), se registran aquí.
-- Permite calcular AUC/LogLoss real y disparar reentrenamiento si baja la calidad.

CREATE TABLE IF NOT EXISTS ground_truth (
    id           BIGSERIAL PRIMARY KEY,
    msno         TEXT     NOT NULL,
    period       TEXT     NOT NULL,
    actual_churn SMALLINT NOT NULL CHECK (actual_churn IN (0, 1)),
    loaded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (msno, period)
);

CREATE INDEX IF NOT EXISTS idx_gt_period ON ground_truth (period);


-- ── DRIFT REPORTS ────────────────────────────────────────────────────────────
-- Resumen de cada ejecución del drift check (Evidently).
-- El detalle completo se guarda como HTML en reports/monitoring/.

CREATE TABLE IF NOT EXISTS drift_reports (
    id              BIGSERIAL PRIMARY KEY,
    period          TEXT    NOT NULL,
    report_date     DATE    NOT NULL,
    dataset_drift   BOOLEAN NOT NULL,
    drift_score     REAL    NOT NULL,
    n_drifted_cols  SMALLINT NOT NULL,
    report_path     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
