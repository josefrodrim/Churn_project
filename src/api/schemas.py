"""
Pydantic schemas — contratos de datos para todo el sistema.

Tres familias:
  FeatureRecord   → representa las 48 features v5 de un usuario
  PredictionRequest/Response → contrato del API
  PredictionRecord → lo que se persiste en la tabla predictions
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── FEATURE RECORD ────────────────────────────────────────────────────────────

class FeatureRecord(BaseModel):
    """48 features v5 de un usuario. Todos los campos son requeridos en inferencia."""

    msno: str = Field(..., description="User ID (KKBox hash)")

    # Transacciones base
    n_transactions:       float = Field(..., ge=0)
    n_cancels:            float = Field(..., ge=0)
    ever_canceled:        int   = Field(..., ge=0, le=1)
    avg_discount_pct:     float = Field(..., ge=0, le=1)
    avg_plan_days:        float = Field(..., ge=0)
    avg_price:            float = Field(..., ge=0)
    n_unique_plans:       float = Field(..., ge=0)
    n_payment_methods:    float = Field(..., ge=0)
    last_is_cancel:       int   = Field(..., ge=0, le=1)
    last_is_auto_renew:   int   = Field(..., ge=0, le=1)
    last_plan_days:       float = Field(..., ge=0)
    last_price:           float = Field(..., ge=0)
    last_list_price:      float = Field(..., ge=0)
    price_trend:          float
    last_payment_method:  int   = Field(..., ge=0)

    # Members
    city:             int   = Field(..., ge=0)
    registered_via:   int   = Field(..., ge=0)
    gender_enc:       int   = Field(..., ge=0, le=2)
    age:              float = Field(..., ge=0, le=120)
    bd_valid:         int   = Field(..., ge=0, le=1)
    tenure_days:      float = Field(..., ge=0)
    has_member_record: int  = Field(..., ge=0, le=1)

    # User logs base
    n_days:              float = Field(..., ge=0)
    avg_daily_secs:      float = Field(..., ge=0)
    avg_daily_completed: float = Field(..., ge=0)
    avg_daily_unq:       float = Field(..., ge=0)
    completion_ratio:    float = Field(..., ge=0, le=1)
    days_since_last:     float = Field(..., ge=0)
    listening_trend:     float
    has_log_record:      int   = Field(..., ge=0, le=1)

    # Expiry + churn lag (v4)
    days_until_expire:    float
    is_expired:           int   = Field(..., ge=0, le=1)
    auto_renew_at_expire: int   = Field(..., ge=0, le=1)
    cancel_at_expire:     int   = Field(..., ge=0, le=1)
    n_renewals:           float = Field(..., ge=0)
    prev_churn:           int   = Field(..., ge=0, le=1)

    # TX recency (v5)
    days_since_last_tx: float = Field(..., ge=0)
    had_tx_last_7d:     int   = Field(..., ge=0, le=1)
    had_tx_last_30d:    int   = Field(..., ge=0, le=1)
    n_tx_last_30d:      float = Field(..., ge=0)

    # Cancel before expire (v5)
    cancel_before_expire: int = Field(..., ge=0, le=1)

    # Multi-window logs (v5)
    n_days_7d:        float = Field(..., ge=0)
    secs_per_day_7d:  float = Field(..., ge=0)
    n_days_90d:       float = Field(..., ge=0)
    secs_per_day_90d: float = Field(..., ge=0)
    trend_7d:         float
    trend_7d_vs_30d:  float

    @field_validator("completion_ratio")
    @classmethod
    def clamp_ratio(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    model_config = {"extra": "forbid"}


# ── API REQUEST / RESPONSE ────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """Predicción individual: acepta el feature vector completo."""
    features: FeatureRecord


class BatchPredictionRequest(BaseModel):
    """Lote de usuarios: máximo 10,000 por request."""
    users: list[FeatureRecord] = Field(..., min_length=1, max_length=10_000)


class PredictionResponse(BaseModel):
    msno:          str
    churn_prob:    float = Field(..., ge=0.0, le=1.0, description="Probabilidad de churn [0,1]")
    churn_label:   int   = Field(..., ge=0, le=1, description="1 si churn_prob > 0.5")
    model_version: str
    predicted_at:  datetime


class BatchPredictionResponse(BaseModel):
    predictions:   list[PredictionResponse]
    model_version: str
    total:         int
    predicted_at:  datetime


# ── PREDICTION RECORD (persistencia) ─────────────────────────────────────────

class PredictionRecord(BaseModel):
    """Lo que se escribe en la tabla predictions de Postgres."""
    msno:          str
    churn_prob:    float
    churn_label:   int
    model_version: str
    predicted_at:  datetime
    period:        str = Field(..., description="Período de predicción, e.g. '2017-04'")
    source:        str = Field(default="api", description="'api' | 'batch'")


# ── MODEL INFO ────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name:          str
    version:       str
    stage:         str
    feature_count: int
    log_loss_val:  Optional[float] = None
    roc_auc_val:   Optional[float] = None
    registered_at: Optional[datetime] = None


# ── HEALTH ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str = "ok"
    model_loaded:  bool
    db_connected:  bool
    model_version: Optional[str] = None
