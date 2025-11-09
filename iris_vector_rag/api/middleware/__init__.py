"""
Middleware components for RAG API.

Provides authentication, rate limiting, and request logging middleware.
"""

from iris_vector_rag.api.middleware.auth import (
    ApiKeyAuth,
    AuthenticationMiddleware
)
from iris_vector_rag.api.middleware.rate_limit import (
    RateLimiter,
    RateLimitMiddleware
)
from iris_vector_rag.api.middleware.logging import (
    RequestLoggingMiddleware,
    MetricsExporter
)


__all__ = [
    "ApiKeyAuth",
    "AuthenticationMiddleware",
    "RateLimiter",
    "RateLimitMiddleware",
    "RequestLoggingMiddleware",
    "MetricsExporter"
]
