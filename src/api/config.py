"""
Configuración centralizada — todas las variables de entorno del sistema.
Nunca hardcodear secrets; todo viene de .env o variables de entorno del contenedor.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── API ───────────────────────────────────────────────────────────────────
    api_host:    str = "0.0.0.0"
    api_port:    int = 8000
    api_key:     str = "changeme"        # X-API-Key header
    debug:       bool = False

    # ── Postgres ──────────────────────────────────────────────────────────────
    postgres_host:     str = "localhost"
    postgres_port:     int = 5432
    postgres_db:       str = "churn"
    postgres_user:     str = "churn_user"
    postgres_password: str = "changeme"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri:  str = "http://localhost:5000"
    mlflow_model_name:    str = "churn-ensemble"
    mlflow_model_stage:   str = "Production"

    # ── Monitoring ────────────────────────────────────────────────────────────
    drift_threshold:       float = 0.15   # dataset drift score que dispara alerta
    min_roc_auc:           float = 0.85   # quality gate para promover modelo
    max_log_loss:          float = 0.240  # quality gate para promover modelo

    # ── Paths ─────────────────────────────────────────────────────────────────
    reports_dir:   str = "reports/monitoring"
    baseline_path: str = "reports/monitoring/baseline.pkl"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Singleton: se instancia una vez y se reutiliza en toda la app."""
    return Settings()
