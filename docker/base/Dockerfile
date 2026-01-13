# =============================================================================
# RAG Templates Base Image
# =============================================================================
# This base image contains common dependencies and configurations shared
# across multiple services in the RAG templates framework.
# =============================================================================

FROM python:3.11-slim as base

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG BUILD_ENV=production

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    # Network and SSL
    ca-certificates \
    openssl \
    # Database drivers
    unixodbc-dev \
    libpq-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    # Text processing
    libxml2-dev \
    libxslt1-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install UV package manager for faster dependency installation
RUN pip install uv

# Create application user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create application directories
WORKDIR /app
RUN mkdir -p /app/logs /app/data /app/cache /app/config \
    && chown -R appuser:appuser /app

# Copy base requirements for common dependencies
COPY docker/base/requirements.txt /app/requirements-base.txt

# Install base Python dependencies
RUN uv pip install --system -r /app/requirements-base.txt

# Copy common utilities and configurations
COPY docker/base/utils/ /app/utils/
COPY docker/base/config/ /app/config/

# Set proper permissions
RUN chown -R appuser:appuser /app

# Health check utility
COPY docker/base/healthcheck.py /usr/local/bin/healthcheck.py
RUN chmod +x /usr/local/bin/healthcheck.py

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /usr/local/bin/healthcheck.py || exit 1

# Switch to non-root user
USER appuser

# Default working directory
WORKDIR /app

# Labels for metadata
LABEL maintainer="RAG Templates Team" \
      version="1.0.0" \
      description="Base image for RAG Templates Framework" \
      org.opencontainers.image.title="RAG Templates Base" \
      org.opencontainers.image.description="Base Docker image with common dependencies" \
      org.opencontainers.image.vendor="RAG Templates Project" \
      org.opencontainers.image.licenses="MIT"