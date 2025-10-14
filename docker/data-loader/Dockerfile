# =============================================================================
# Data Loader Service Dockerfile
# =============================================================================
# Service for initializing the RAG system with sample data and running
# data ingestion pipelines during container startup
# =============================================================================

FROM python:3.11-slim as base

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create data loader user
RUN groupadd -r dataloader && useradd -r -g dataloader dataloader

# Create application directory
WORKDIR /app

# Copy requirements
COPY docker/data-loader/requirements.txt .

# Install UV for faster package installation
RUN pip install uv

# Install Python dependencies
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY docker/data-loader/scripts/ ./scripts/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY rag_templates/ ./rag_templates/
COPY common/ ./common/
COPY adapters/ ./adapters/

# Copy configuration
COPY docker/data-loader/config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/cache \
    && chown -R dataloader:dataloader /app

# Copy entrypoint script
COPY docker/data-loader/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Switch to non-root user
USER dataloader

# Environment variables
ENV PYTHONPATH=/app \
    DATA_LOADER_MODE=batch \
    BATCH_SIZE=100 \
    LOG_LEVEL=INFO

# Default command
ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "scripts/load_sample_data.py"]

# Labels
LABEL maintainer="RAG Templates Team" \
      version="1.0.0" \
      description="Data Loader Service for RAG Templates" \
      org.opencontainers.image.title="RAG Data Loader" \
      org.opencontainers.image.description="Service for loading sample data and running data pipelines" \
      org.opencontainers.image.vendor="RAG Templates Project" \
      org.opencontainers.image.licenses="MIT"