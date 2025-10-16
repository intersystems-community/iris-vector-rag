"""
WebSocket models for RAG API.

Pydantic models for WebSocket Session entity (Entity 7 from data-model.md).
Implements JSON-based event streaming protocol.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from uuid import UUID
from datetime import datetime


class SubscriptionType(str, Enum):
    """WebSocket event subscription types."""

    QUERY_STREAMING = "query_streaming"
    DOCUMENT_UPLOAD = "document_upload"
    ALL = "all"


class WebSocketSession(BaseModel):
    """
    WebSocket Session (Entity 7 from data-model.md).

    Represents an active WebSocket connection for streaming responses.
    Manages message queue, reconnection state, and client subscription preferences.

    Implements FR-025 to FR-028: WebSocket streaming with JSON event protocol.
    """

    session_id: UUID = Field(
        ...,
        description="Unique session identifier",
    )

    api_key_id: UUID = Field(
        ...,
        description="Authenticated API key",
    )

    connected_at: datetime = Field(
        ...,
        description="When connection was established (ISO8601)",
    )

    last_activity_at: datetime = Field(
        ...,
        description="Last message sent/received (ISO8601)",
    )

    client_ip: str = Field(
        ...,
        description="Client IP address",
        examples=["192.168.1.100"],
    )

    subscription_type: SubscriptionType = Field(
        ...,
        description="Event subscriptions",
        examples=["query_streaming"],
    )

    is_active: bool = Field(
        default=True,
        description="Whether connection is open",
    )

    message_count: int = Field(
        default=0,
        ge=0,
        description="Total messages sent",
    )

    reconnection_token: Optional[str] = Field(
        default=None,
        description="Token for reconnecting to session",
        examples=["reconnect_d4e5f6a7-8901-2345-6789-abcdef012345"],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "session_id": "c1d2e3f4-5678-90ab-cdef-123456789abc",
                "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "connected_at": "2025-10-16T12:34:00.000Z",
                "last_activity_at": "2025-10-16T12:34:56.789Z",
                "client_ip": "192.168.1.100",
                "subscription_type": "query_streaming",
                "is_active": True,
                "message_count": 23,
                "reconnection_token": "reconnect_d4e5f6a7-8901-2345-6789-abcdef012345",
            }
        }


class EventType(str, Enum):
    """WebSocket event types."""

    # Query streaming events
    QUERY_START = "query_start"
    RETRIEVAL_PROGRESS = "retrieval_progress"
    GENERATION_CHUNK = "generation_chunk"
    QUERY_COMPLETE = "query_complete"

    # Document upload events
    DOCUMENT_UPLOAD_PROGRESS = "document_upload_progress"

    # Generic events
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


class WebSocketEvent(BaseModel):
    """
    WebSocket event message.

    Implements FR-028: JSON-based event streaming protocol.
    All events follow this structure.
    """

    event: EventType = Field(
        ...,
        description="Event type",
        examples=["query_start"],
    )

    data: Dict[str, Any] = Field(
        ...,
        description="Event-specific data",
    )

    timestamp: datetime = Field(
        ...,
        description="Event timestamp (ISO8601)",
    )

    request_id: UUID = Field(
        ...,
        description="Associated request identifier",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "event": "document_upload_progress",
                "data": {
                    "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
                    "processed_documents": 47,
                    "total_documents": 100,
                    "progress_percentage": 47.0,
                },
                "timestamp": "2025-10-16T12:30:15.000Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }


class DocumentUploadProgressEvent(BaseModel):
    """
    Document upload progress event data.

    Implements FR-027: Stream document loading progress with percentage.
    """

    operation_id: UUID = Field(
        ...,
        description="Document upload operation ID",
    )

    processed_documents: int = Field(
        ...,
        ge=0,
        description="Documents processed so far",
    )

    total_documents: int = Field(
        ...,
        gt=0,
        description="Total documents to process",
    )

    progress_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Completion percentage (0.0-100.0)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
                "processed_documents": 47,
                "total_documents": 100,
                "progress_percentage": 47.0,
            }
        }


class QueryProgressEvent(BaseModel):
    """
    Query streaming progress event data.

    Implements FR-026: Stream incremental query results.
    """

    query: str = Field(
        ...,
        description="Original query text",
    )

    pipeline: str = Field(
        ...,
        description="Pipeline processing the query",
    )

    documents_retrieved: Optional[int] = Field(
        default=None,
        description="Number of documents retrieved so far",
    )

    generation_chunk: Optional[str] = Field(
        default=None,
        description="Partial generated answer text",
    )

    is_final: bool = Field(
        default=False,
        description="Whether this is the final chunk",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "query": "What is diabetes?",
                "pipeline": "graphrag",
                "documents_retrieved": 5,
                "generation_chunk": "Diabetes is a chronic metabolic disorder...",
                "is_final": False,
            }
        }


class WebSocketAuthMessage(BaseModel):
    """
    WebSocket authentication message.

    Client sends this as first message after connecting.
    """

    api_key: str = Field(
        ...,
        description="Base64-encoded API key (id:secret)",
    )

    subscription_type: SubscriptionType = Field(
        default=SubscriptionType.ALL,
        description="Which events to subscribe to",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "api_key": "N2M5ZTY2NzktNzQyNS00MGRlLTk0NGItZTA3ZmMxZjkwYWU3Om15X3NlY3JldF9rZXlfMTIzNDU=",
                "subscription_type": "query_streaming",
            }
        }
