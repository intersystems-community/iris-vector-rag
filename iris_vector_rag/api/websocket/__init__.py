"""
WebSocket connection handlers and event streaming.

Provides real-time streaming for query execution and document upload progress.
"""

from iris_vector_rag.api.websocket.connection import WebSocketConnectionManager
from iris_vector_rag.api.websocket.handlers import (
    DocumentUploadProgressHandler,
    QueryStreamingHandler,
)
from iris_vector_rag.api.websocket.routes import create_websocket_router

__all__ = [
    "WebSocketConnectionManager",
    "QueryStreamingHandler",
    "DocumentUploadProgressHandler",
    "create_websocket_router",
]
