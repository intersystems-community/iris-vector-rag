"""
REST API for RAG Pipelines.

Production-grade FastAPI server providing HTTP endpoints for querying RAG pipelines,
managing pipeline lifecycle, document uploads, and WebSocket streaming.

Features:
- API key authentication (Elasticsearch-inspired)
- Adaptive rate limiting
- WebSocket streaming for long-running operations
- OpenAPI 3.1 specification
- IRIS connection pooling
"""

import warnings

warnings.warn(
    "iris_vector_rag.api (the REST API) is deprecated in 0.15 and will be removed "
    "in 0.16. It cannot serve a request as built. Use the MCP server "
    "(iris_vector_rag.mcp) or call the pipelines from Python. The code and its "
    "tests stay at the git tag archive/rest-api-v1; the design is kept in "
    "docs/archived/rest-api/.",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = "0.1.0"
