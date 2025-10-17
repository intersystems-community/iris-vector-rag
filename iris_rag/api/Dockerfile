# =============================================================================
# Multi-Stage Dockerfile for RAG API Server
# =============================================================================
# Implements FR-032: Production-ready Docker deployment
# Optimized for security, performance, and minimal image size
# =============================================================================

# =============================================================================
# Stage 1: Build Stage
# =============================================================================
FROM python:3.11-slim AS builder

# Set build arguments
ARG UV_VERSION=0.1.18

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./
COPY common/ ./common/
COPY iris_rag/ ./iris_rag/
COPY config/ ./config/

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install dependencies
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Runtime Stage
# =============================================================================
FROM python:3.11-slim AS runtime

# Set runtime arguments
ARG API_USER=apiuser
ARG API_UID=1000
ARG API_GID=1000

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -g ${API_GID} ${API_USER} && \
    useradd -m -u ${API_UID} -g ${API_GID} -s /bin/bash ${API_USER}

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=${API_USER}:${API_USER} common/ ./common/
COPY --chown=${API_USER}:${API_USER} iris_rag/ ./iris_rag/
COPY --chown=${API_USER}:${API_USER} config/ ./config/
COPY --chown=${API_USER}:${API_USER} pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/logs /app/outputs && \
    chown -R ${API_USER}:${API_USER} /app/logs /app/outputs

# Set environment variables
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:${PYTHONPATH}"

# Expose API port
EXPOSE 8000

# Switch to non-root user
USER ${API_USER}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command: Run API server
CMD ["python", "-m", "iris_rag.api.cli", "run", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
