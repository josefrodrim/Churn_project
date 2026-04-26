# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Instala dependencias Python. Separado para que el cache de pip no se invalide
# cada vez que cambia código fuente.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Imagen final slim — solo lo necesario para correr el API.
FROM python:3.11-slim AS runtime

# libpq5: cliente Postgres en runtime (psycopg2 lo necesita)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copiar deps instaladas del builder al home del usuario
COPY --from=builder /root/.local /home/appuser/.local

# Solo el código fuente — sin datos, notebooks ni modelos
COPY src/ ./src/

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--access-log"]
