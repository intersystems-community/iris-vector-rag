# =============================================================================
# Nginx Reverse Proxy Dockerfile
# =============================================================================
# Production-ready reverse proxy with SSL termination, load balancing,
# and optimized configurations for the RAG Templates framework
# =============================================================================

FROM nginx:alpine as base

# Install additional packages
RUN apk add --no-cache \
    curl \
    openssl \
    certbot \
    certbot-nginx \
    logrotate

# Create nginx user and directories
RUN addgroup -g 101 -S nginx || true && \
    adduser -S -D -H -u 101 -h /var/cache/nginx -s /sbin/nologin -G nginx -g nginx nginx || true

# Create necessary directories
RUN mkdir -p /var/log/nginx /var/cache/nginx /etc/nginx/ssl /etc/nginx/conf.d \
    && chown -R nginx:nginx /var/log/nginx /var/cache/nginx /etc/nginx/ssl

# Copy custom nginx configuration
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker/nginx/conf.d/ /etc/nginx/conf.d/
COPY docker/nginx/ssl/ /etc/nginx/ssl/

# Copy startup script
COPY docker/nginx/entrypoint.sh /docker-entrypoint.d/99-rag-setup.sh
RUN chmod +x /docker-entrypoint.d/99-rag-setup.sh

# Create self-signed SSL certificate for development
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/nginx-selfsigned.key \
    -out /etc/nginx/ssl/nginx-selfsigned.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"

# Copy health check script
COPY docker/nginx/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Expose ports
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

# Use custom entrypoint that handles configuration
ENTRYPOINT ["/docker-entrypoint.sh"]

# Switch to non-root user
USER nginx

# Default command
CMD ["nginx", "-g", "daemon off;"]

# Production stage with additional security
FROM base as production

# Remove unnecessary packages and clean up
RUN apk del --no-cache \
    && rm -rf /var/cache/apk/* \
    && rm -rf /tmp/*

# Set stricter permissions
RUN chmod 600 /etc/nginx/ssl/nginx-selfsigned.key \
    && chmod 644 /etc/nginx/ssl/nginx-selfsigned.crt

# Labels
LABEL maintainer="RAG Templates Team" \
      version="1.0.0" \
      description="Nginx Reverse Proxy for RAG Templates" \
      org.opencontainers.image.title="RAG Nginx Proxy" \
      org.opencontainers.image.description="Optimized Nginx reverse proxy with SSL termination" \
      org.opencontainers.image.vendor="RAG Templates Project" \
      org.opencontainers.image.licenses="MIT"