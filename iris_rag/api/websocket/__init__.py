"""
WebSocket connection handlers and event streaming.

Provides real-time streaming for query execution and document upload progress.
"""

from iris_rag.api.websocket.connection import ConnectionManager
from iris_rag.api.websocket.handlers import (
    QueryStreamingHandler,
    DocumentUploadProgressHandler
)
from iris_rag.api.websocket.routes import create_websocket_router


__all__ = [
    "ConnectionManager",
    "QueryStreamingHandler",
    "DocumentUploadProgressHandler",
    "create_websocket_router"
]
