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

__version__ = "0.1.0"
