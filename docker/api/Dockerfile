# =============================================================================
# RAG API Service Dockerfile
# =============================================================================
# Multi-stage production-optimized container for the RAG API service
# Provides RESTful endpoints for all RAG pipeline operations
# =============================================================================

# Build stage
FROM python:3.11-slim as builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG BUILD_ENV=production

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy requirements files
COPY requirements.txt .
COPY docker/api/requirements.txt ./requirements-api.txt

# Install UV for faster package installation
RUN pip install uv

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN uv pip install --no-cache -r requirements.txt
RUN uv pip install --no-cache -r requirements-api.txt

# Production stage
FROM python:3.11-slim as production

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    FASTAPI_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WORKERS=4

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r apiuser && useradd -r -g apiuser apiuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY rag_templates/ ./rag_templates/
COPY common/ ./common/
COPY adapters/ ./adapters/
COPY docker/api/app/ ./
COPY docker/api/entrypoint.sh ./entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache /app/config \
    && chown -R apiuser:apiuser /app \
    && chmod +x /app/entrypoint.sh

# Copy health check utility
COPY docker/base/healthcheck.py /usr/local/bin/healthcheck.py
RUN chmod +x /usr/local/bin/healthcheck.py

# Switch to non-root user
USER apiuser

# Expose API port
EXPOSE 8000

# Health check is handled by docker-compose.yml using curl
# HEALTHCHECK disabled in favor of compose-level healthcheck

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Development stage
FROM production as development

# Switch back to root for development packages
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
COPY docker/api/requirements-dev.txt ./requirements-dev.txt
RUN uv pip install --no-cache -r requirements-dev.txt

# Set development environment
ENV FASTAPI_ENV=development \
    API_DEBUG=true \
    API_RELOAD=true

# Switch back to apiuser
USER apiuser

# Development command with auto-reload
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]

# Labels
LABEL maintainer="RAG Templates Team" \
      version="1.0.0" \
      description="RAG API Service - RESTful endpoints for RAG pipelines" \
      org.opencontainers.image.title="RAG API Service" \
      org.opencontainers.image.description="FastAPI service providing RAG pipeline endpoints" \
      org.opencontainers.image.vendor="RAG Templates Project" \
      org.opencontainers.image.licenses="MIT"