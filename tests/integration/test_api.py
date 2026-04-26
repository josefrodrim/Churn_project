"""
Tests de integración — endpoints del API.

Usa MockModelManager (sin cargar modelos reales) y sin DB real.
Cubre: health, predict individual, predict batch, autenticación y errores.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from tests.conftest import BASE_FEATURES


# ── HEALTH ────────────────────────────────────────────────────────────────────

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        # DB mockeada → db_connected=False, pero modelo sí está cargado
        with patch("src.api.routes.admin.get_db") as mock_db:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.return_value.__aexit__  = AsyncMock(return_value=False)

            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_health_has_model_version(self, client):
        with patch("src.api.routes.admin.get_db") as mock_db:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_db.return_value.__aexit__  = AsyncMock(return_value=False)

            resp = await client.get("/health")

        assert resp.json()["model_version"] == "test-v0"


# ── PREDICT INDIVIDUAL ────────────────────────────────────────────────────────

class TestPredictSingle:

    def _payload(self, msno: str = "user_abc", overrides: dict = None) -> dict:
        features = {**BASE_FEATURES, **(overrides or {})}
        return {"features": {"msno": msno, **features}}

    @pytest.mark.asyncio
    async def test_predict_requires_api_key(self, client):
        resp = await client.post("/predict", json=self._payload())
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_predict_valid_request(self, client):
        with patch("src.api.routes.predict.log_prediction", new_callable=AsyncMock):
            resp = await client.post(
                "/predict",
                json=self._payload(),
                headers={"X-API-Key": "changeme"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "churn_prob" in data
        assert 0.0 <= data["churn_prob"] <= 1.0
        assert data["churn_label"] in (0, 1)
        assert data["msno"] == "user_abc"
        assert data["model_version"] == "test-v0"

    @pytest.mark.asyncio
    async def test_predict_expired_user_high_prob(self, client):
        with patch("src.api.routes.predict.log_prediction", new_callable=AsyncMock):
            resp = await client.post(
                "/predict",
                json=self._payload("expired_user", {"is_expired": 1}),
                headers={"X-API-Key": "changeme"},
            )
        assert resp.status_code == 200
        assert resp.json()["churn_prob"] > 0.5
        assert resp.json()["churn_label"] == 1

    @pytest.mark.asyncio
    async def test_predict_active_user_low_prob(self, client):
        with patch("src.api.routes.predict.log_prediction", new_callable=AsyncMock):
            resp = await client.post(
                "/predict",
                json=self._payload("active_user", {"is_expired": 0}),
                headers={"X-API-Key": "changeme"},
            )
        assert resp.status_code == 200
        assert resp.json()["churn_prob"] < 0.5
        assert resp.json()["churn_label"] == 0

    @pytest.mark.asyncio
    async def test_predict_missing_feature_returns_422(self, client):
        incomplete = {"features": {"msno": "x", "n_transactions": 1.0}}
        resp = await client.post(
            "/predict",
            json=incomplete,
            headers={"X-API-Key": "changeme"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_wrong_api_key_returns_401(self, client):
        resp = await client.post(
            "/predict",
            json=self._payload(),
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401


# ── PREDICT BATCH ────────────────────────────────────────────────────────────

class TestPredictBatch:

    def _batch_payload(self, n: int = 3) -> dict:
        users = [
            {"msno": f"user_{i}", **BASE_FEATURES}
            for i in range(n)
        ]
        return {"users": users}

    @pytest.mark.asyncio
    async def test_batch_predict_returns_all_users(self, client):
        with patch("src.api.routes.predict.log_prediction", new_callable=AsyncMock):
            resp = await client.post(
                "/predict/batch",
                json=self._batch_payload(5),
                headers={"X-API-Key": "changeme"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["predictions"]) == 5

    @pytest.mark.asyncio
    async def test_batch_predict_probabilities_valid(self, client):
        with patch("src.api.routes.predict.log_prediction", new_callable=AsyncMock):
            resp = await client.post(
                "/predict/batch",
                json=self._batch_payload(3),
                headers={"X-API-Key": "changeme"},
            )
        for pred in resp.json()["predictions"]:
            assert 0.0 <= pred["churn_prob"] <= 1.0

    @pytest.mark.asyncio
    async def test_batch_empty_list_returns_422(self, client):
        resp = await client.post(
            "/predict/batch",
            json={"users": []},
            headers={"X-API-Key": "changeme"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_requires_api_key(self, client):
        resp = await client.post("/predict/batch", json=self._batch_payload())
        assert resp.status_code == 401


# ── QUALITY GATE ──────────────────────────────────────────────────────────────

class TestQualityGate:
    """
    Valida que el MockModelManager cumple el threshold mínimo de calidad.
    En CI, aquí iría el modelo real contra un sample dataset precacheado.
    """

    @pytest.mark.asyncio
    async def test_expired_users_predicted_as_churn(self, client):
        """Usuarios expirados deben tener churn_prob > 0.5 — invariante de negocio."""
        expired_payload = {
            "users": [
                {"msno": f"expired_{i}", **BASE_FEATURES, "is_expired": 1}
                for i in range(10)
            ]
        }
        with patch("src.api.routes.predict.log_prediction", new_callable=AsyncMock):
            resp = await client.post(
                "/predict/batch",
                json=expired_payload,
                headers={"X-API-Key": "changeme"},
            )
        predictions = resp.json()["predictions"]
        churn_rate = sum(p["churn_label"] for p in predictions) / len(predictions)
        assert churn_rate == 1.0, f"Todos los expirados deben ser churn=1, got {churn_rate:.0%}"
