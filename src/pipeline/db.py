"""
Utilidades de conexión a Postgres — compartidas por pipeline y API.

Ofrece:
  get_engine()      → SQLAlchemy Engine (sync, para pipelines batch)
  get_async_engine() → AsyncEngine (para FastAPI, se crea en Fase 3)
"""

from functools import lru_cache

from sqlalchemy import create_engine, text, Engine

from src.api.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = get_settings()
    url = (
        f"postgresql+psycopg2://{cfg.postgres_user}:{cfg.postgres_password}"
        f"@{cfg.postgres_host}:{cfg.postgres_port}/{cfg.postgres_db}"
    )
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)


def check_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
