"""Middleware components for RAG API."""

def __getattr__(name):
    if name in ("ApiKeyAuth", "AuthenticationMiddleware"):
        from iris_vector_rag.api.middleware.auth import ApiKeyAuth, AuthenticationMiddleware
        return locals()[name]
    if name in ("RateLimiter", "RateLimitMiddleware"):
        from iris_vector_rag.api.middleware.rate_limit import RateLimiter, RateLimitMiddleware
        return locals()[name]
    if name in ("RequestLoggingMiddleware", "MetricsExporter"):
        from iris_vector_rag.api.middleware.logging import RequestLoggingMiddleware, MetricsExporter
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ApiKeyAuth",
    "AuthenticationMiddleware",
    "RateLimiter",
    "RateLimitMiddleware",
    "RequestLoggingMiddleware",
    "MetricsExporter",
]
