# ═══════════════════════════════════════════════════════════════════════════
# LogSentinel AI — Multi-Stage Production Dockerfile
# Stage 1: builder  — compile dependencies, download models
# Stage 2: production — minimal runtime image
# ═══════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

LABEL maintainer="LogSentinel AI <team@logsentinel.ai>"
LABEL version="1.0.0"
LABEL description="Real-Time Anomaly Detection in System Logs"

# Build-time dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies into a prefix directory
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Production ───────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS production

ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION=1.0.0

LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.title="LogSentinel AI"
LABEL org.opencontainers.image.description="Real-Time Anomaly Detection in System Logs"
LABEL org.opencontainers.image.licenses="MIT"

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r logsentinel && \
    useradd -r -g logsentinel -d /app -s /bin/bash -c "LogSentinel AI" logsentinel

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy application source
COPY --chown=logsentinel:logsentinel backend/ ./backend/
COPY --chown=logsentinel:logsentinel ml_models/ ./ml_models/
COPY --chown=logsentinel:logsentinel kafka/ ./kafka/
COPY --chown=logsentinel:logsentinel datasets/ ./datasets/
COPY --chown=logsentinel:logsentinel scripts/ ./scripts/
COPY --chown=logsentinel:logsentinel config/ ./config/
COPY --chown=logsentinel:logsentinel monitoring/ ./monitoring/
COPY --chown=logsentinel:logsentinel alembic.ini ./
COPY --chown=logsentinel:logsentinel alembic/ ./alembic/

# Create runtime directories
RUN mkdir -p \
    /app/ml_models/saved \
    /app/reports/generated \
    /app/logs \
    && chown -R logsentinel:logsentinel /app

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    WORKERS=4 \
    LOG_LEVEL=INFO \
    ENVIRONMENT=production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/api/v1/health/live || exit 1

# Switch to non-root user
USER logsentinel

# Use tini as PID 1 (handles zombie processes)
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: production uvicorn with gunicorn process manager
CMD ["sh", "-c", "uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level ${LOG_LEVEL,,} \
    --no-access-log \
    --proxy-headers \
    --forwarded-allow-ips='*'"]

# ── Stage 3: Development ─────────────────────────────────────────────────────
FROM production AS development

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

USER logsentinel

CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--log-level", "debug"]
