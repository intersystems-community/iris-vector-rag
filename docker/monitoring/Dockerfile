# =============================================================================
# Health Monitoring Service Dockerfile
# =============================================================================
# Lightweight monitoring service for tracking system health, metrics,
# and alerting across the RAG Templates framework
# =============================================================================

FROM python:3.11-alpine as base

# Build arguments
ARG MONITORING_USER=monitor
ARG MONITORING_UID=1001
ARG MONITORING_GID=1001

# Install system dependencies
RUN apk add --no-cache \
    curl \
    bash \
    docker-cli \
    && rm -rf /var/cache/apk/*

# Create monitoring user
RUN addgroup -g ${MONITORING_GID} ${MONITORING_USER} && \
    adduser -D -u ${MONITORING_UID} -G ${MONITORING_USER} ${MONITORING_USER}

# Create application directory
WORKDIR /app

# Copy requirements
COPY docker/monitoring/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY docker/monitoring/app/ ./
COPY docker/monitoring/config/ ./config/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/alerts \
    && chown -R ${MONITORING_USER}:${MONITORING_USER} /app

# Copy health check utility
COPY docker/base/healthcheck.py /usr/local/bin/healthcheck.py
RUN chmod +x /usr/local/bin/healthcheck.py

# Switch to monitoring user
USER ${MONITORING_USER}

# Expose monitoring port
EXPOSE 9090

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python /usr/local/bin/healthcheck.py --service monitoring --http http://localhost:9090/health

# Default command
CMD ["python", "monitor.py"]

# Labels
LABEL maintainer="RAG Templates Team" \
      version="1.0.0" \
      description="Health Monitoring Service for RAG Templates" \
      org.opencontainers.image.title="RAG Monitoring Service" \
      org.opencontainers.image.description="System health monitoring and alerting" \
      org.opencontainers.image.vendor="RAG Templates Project" \
      org.opencontainers.image.licenses="MIT"