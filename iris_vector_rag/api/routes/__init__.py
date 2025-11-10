"""
API route modules for RAG API.

Provides query, pipeline, document, and health check endpoints.
"""

from iris_vector_rag.api.routes.query import create_query_router
from iris_vector_rag.api.routes.pipeline import create_pipeline_router
from iris_vector_rag.api.routes.document import create_document_router
from iris_vector_rag.api.routes.health import create_health_router


__all__ = [
    "create_query_router",
    "create_pipeline_router",
    "create_document_router",
    "create_health_router"
]
