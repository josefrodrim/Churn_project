"""
Fixtures compartidos entre tests unitarios e de integración.

- dummy_features   : FeatureRecord válido con valores representativos
- edge_cases       : casos borde (sin logs, expirado, primer mes)
- mock_model       : modelo dummy que devuelve prob fija (evita cargar 3 modelos)
- app_client       : AsyncClient del API con modelo mock inyectado
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.api.schemas import FeatureRecord
from src.models.retrain_submit_14 import FEATURE_COLS_V5


# ── FEATURE RECORD BASE ───────────────────────────────────────────────────────

BASE_FEATURES = {
    # transacciones
    "n_transactions": 12.0, "n_cancels": 1.0, "ever_canceled": 1,
    "avg_discount_pct": 0.05, "avg_plan_days": 30.0, "avg_price": 149.0,
    "n_unique_plans": 2.0, "n_payment_methods": 1.0,
    "last_is_cancel": 0, "last_is_auto_renew": 1,
    "last_plan_days": 30.0, "last_price": 149.0, "last_list_price": 149.0,
    "price_trend": 0.0, "last_payment_method": 36,
    # members
    "city": 1, "registered_via": 4, "gender_enc": 0,
    "age": 28.0, "bd_valid": 1, "tenure_days": 365.0, "has_member_record": 1,
    # logs
    "n_days": 20.0, "avg_daily_secs": 3600.0, "avg_daily_completed": 10.0,
    "avg_daily_unq": 8.0, "completion_ratio": 0.75,
    "days_since_last": 2.0, "listening_trend": 0.1, "has_log_record": 1,
    # expiry
    "days_until_expire": 15.0, "is_expired": 0,
    "auto_renew_at_expire": 1, "cancel_at_expire": 0,
    "n_renewals": 11.0, "prev_churn": 0,
    # tx recency
    "days_since_last_tx": 5.0, "had_tx_last_7d": 1,
    "had_tx_last_30d": 1, "n_tx_last_30d": 1.0,
    # cancel before expire + multiwindow logs
    "cancel_before_expire": 0,
    "n_days_7d": 5.0, "secs_per_day_7d": 3200.0,
    "n_days_90d": 60.0, "secs_per_day_90d": 3400.0,
    "trend_7d": -100.0, "trend_7d_vs_30d": -50.0,
}


@pytest.fixture
def dummy_features() -> FeatureRecord:
    """Usuario activo típico — churn poco probable."""
    return FeatureRecord(msno="test_user_active", **BASE_FEATURES)


@pytest.fixture
def edge_no_logs() -> FeatureRecord:
    """Usuario sin historial de logs (nuevo o inactivo total)."""
    f = {**BASE_FEATURES,
         "n_days": 0.0, "avg_daily_secs": 0.0, "avg_daily_completed": 0.0,
         "avg_daily_unq": 0.0, "completion_ratio": 0.0,
         "days_since_last": 999.0, "listening_trend": 0.0, "has_log_record": 0,
         "n_days_7d": 0.0, "secs_per_day_7d": 0.0,
         "n_days_90d": 0.0, "secs_per_day_90d": 0.0,
         "trend_7d": 0.0, "trend_7d_vs_30d": 0.0}
    return FeatureRecord(msno="test_user_no_logs", **f)


@pytest.fixture
def edge_expired() -> FeatureRecord:
    """Membresía vencida + canceló antes de expirar — churn casi seguro."""
    f = {**BASE_FEATURES,
         "days_until_expire": -30.0, "is_expired": 1,
         "auto_renew_at_expire": 0, "cancel_at_expire": 1,
         "last_is_cancel": 1, "cancel_before_expire": 1,
         "prev_churn": 1, "days_since_last": 45.0}
    return FeatureRecord(msno="test_user_expired", **f)


@pytest.fixture
def edge_first_month() -> FeatureRecord:
    """Usuario en su primer mes — sin historial previo."""
    f = {**BASE_FEATURES,
         "n_transactions": 1.0, "n_cancels": 0.0, "ever_canceled": 0,
         "tenure_days": 15.0, "n_renewals": 0.0, "prev_churn": 0,
         "had_tx_last_7d": 1, "had_tx_last_30d": 1, "n_tx_last_30d": 1.0}
    return FeatureRecord(msno="test_user_new", **f)


# ── MOCK MODEL ────────────────────────────────────────────────────────────────

class MockModelManager:
    """Reemplaza ModelManager en tests — no carga ningún modelo real."""
    is_loaded    = True
    version      = "test-v0"
    source       = "mock"

    def predict(self, df):
        import numpy as np
        # Retorna prob alta para usuarios con is_expired=1, baja para el resto
        expired = df["is_expired"].values if "is_expired" in df.columns else [0]
        return np.where(expired == 1, 0.85, 0.12).astype(float)


@pytest.fixture
def mock_model(monkeypatch):
    """Inyecta MockModelManager en las dependencias del API."""
    from src.api import dependencies
    mock = MockModelManager()
    monkeypatch.setattr(dependencies, "model_manager", mock)
    return mock


# ── APP CLIENT ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client(mock_model, monkeypatch):
    """
    AsyncClient del API con:
      - Modelo mock inyectado (no carga LightGBM/XGB/CatBoost)
      - API_KEY conocida para los tests
      - DB mockeada (no necesita Postgres real)
    """
    import unittest.mock as mock_lib
    from src.api.app import create_app
    from src.api import dependencies
    from src.api.config import get_settings

    # Fijar API_KEY conocida y limpiar el singleton cacheado
    monkeypatch.setenv("API_KEY", "changeme")
    get_settings.cache_clear()

    # Mock de sesión async que soporta `async with`
    mock_session = mock_lib.AsyncMock()
    mock_session.__aenter__ = mock_lib.AsyncMock(return_value=mock_session)
    mock_session.__aexit__  = mock_lib.AsyncMock(return_value=False)
    mock_session_factory    = mock_lib.MagicMock(return_value=mock_session)

    with mock_lib.patch.object(dependencies, "init_db", return_value=None), \
         mock_lib.patch.object(dependencies, "_AsyncSessionLocal",
                               mock_session_factory, create=True):
        app       = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    # Restaurar cache para que no afecte otros tests
    get_settings.cache_clear()
