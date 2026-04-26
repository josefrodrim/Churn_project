"""
FastAPI application — entry point.

Levanta el servidor con:
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

O desde Docker Compose (Fase 4).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.config import get_settings
from src.api.dependencies import close_db, init_db, model_manager
from src.api.routes.admin import router as admin_router
from src.api.routes.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── LIFESPAN: startup / shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    log.info("── Startup ──────────────────────────────────────")

    # 1. Inicializar DB async
    try:
        init_db(cfg)
        log.info("DB async engine inicializado")
    except Exception as e:
        log.warning(f"DB no disponible en startup: {e}")

    # 2. Cargar modelo (MLflow → fallback joblib)
    log.info("Cargando modelo...")
    model_manager.load(cfg)
    log.info(f"Modelo listo: v{model_manager.version} (fuente: {model_manager.source})")

    log.info("── API lista ────────────────────────────────────")
    yield

    # Shutdown
    log.info("── Shutdown ─────────────────────────────────────")
    await close_db()
    log.info("DB cerrada")


# ── APP ───────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title       = "KKBox Churn Prediction API",
        description = (
            "Predice la probabilidad de churn de usuarios de KKBox. "
            "Ensemble de LightGBM + XGBoost + CatBoost (48 features v5). "
            "Autenticación: header `X-API-Key`."
        ),
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        lifespan    = lifespan,
        debug       = cfg.debug,
    )

    # CORS — solo necesario si hay un frontend separado
    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["GET", "POST"],
        allow_headers  = ["*"],
    )

    # Routers
    app.include_router(admin_router)
    app.include_router(predict_router)

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/docs")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host    = cfg.api_host,
        port    = cfg.api_port,
        reload  = cfg.debug,
        workers = 1,
    )
