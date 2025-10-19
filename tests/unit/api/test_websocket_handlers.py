"""
Unit tests for WebSocket handlers.

Tests the WebSocket connection and event handlers in isolation.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iris_rag.api.models.auth import ApiKey, Permission, RateLimitTier
from iris_rag.api.models.websocket import WebSocketEvent, WebSocketSession
from iris_rag.api.websocket.connection import ConnectionManager


class TestConnectionManager:
    """Test WebSocket connection manager."""

    @pytest.fixture
    def connection_manager(self):
        """Create connection manager instance."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        ws = MagicMock()
        ws.send_json = AsyncMock()
        ws.receive_json = AsyncMock()
        ws.close = AsyncMock()
        return ws

    @pytest.fixture
    def test_api_key(self):
        """Create test API key."""
        return ApiKey(
            key_id="test-key-id",
            key_secret_hash="hashed-secret",
            name="Test Key",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="test@example.com",
        )

    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager, mock_websocket, test_api_key):
        """Test connecting WebSocket client."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="all",
        )

        assert session_id is not None
        assert len(connection_manager.active_connections) == 1

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, connection_manager, mock_websocket, test_api_key):
        """Test disconnecting WebSocket client."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="all",
        )

        await connection_manager.disconnect(session_id)

        assert len(connection_manager.active_connections) == 0

    @pytest.mark.asyncio
    async def test_send_message_to_session(self, connection_manager, mock_websocket, test_api_key):
        """Test sending message to specific session."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="all",
        )

        event = WebSocketEvent(
            event="test_event",
            data={"message": "Hello"},
            timestamp=datetime.utcnow(),
        )

        await connection_manager.send_to_session(session_id, event)

        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, connection_manager, test_api_key):
        """Test broadcasting message to all connections."""
        # Connect multiple clients
        ws1 = MagicMock()
        ws1.send_json = AsyncMock()
        ws2 = MagicMock()
        ws2.send_json = AsyncMock()

        await connection_manager.connect(ws1, test_api_key, "all")
        await connection_manager.connect(ws2, test_api_key, "all")

        event = WebSocketEvent(
            event="broadcast",
            data={"message": "Hello all"},
            timestamp=datetime.utcnow(),
        )

        await connection_manager.broadcast(event)

        ws1.send_json.assert_called_once()
        ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_heartbeat(self, connection_manager, mock_websocket, test_api_key):
        """Test sending heartbeat ping."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="all",
        )

        await connection_manager.send_heartbeat(session_id)

        mock_websocket.send_json.assert_called()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["event"] == "ping"

    @pytest.mark.asyncio
    async def test_max_connections_per_key(self, connection_manager, test_api_key):
        """Test maximum connections per API key limit."""
        connections = []

        # Try to create more connections than allowed
        for i in range(15):  # Max is usually 10
            ws = MagicMock()
            ws.send_json = AsyncMock()
            ws.close = AsyncMock()

            try:
                session_id = await connection_manager.connect(
                    websocket=ws,
                    api_key=test_api_key,
                    subscription_type="all",
                )
                connections.append(session_id)
            except Exception:
                break

        # Should not exceed limit
        assert len(connections) <= 10

    @pytest.mark.asyncio
    async def test_get_active_session_count(self, connection_manager, mock_websocket, test_api_key):
        """Test getting active session count."""
        assert connection_manager.get_active_session_count() == 0

        await connection_manager.connect(mock_websocket, test_api_key, "all")

        assert connection_manager.get_active_session_count() == 1

    @pytest.mark.asyncio
    async def test_cleanup_inactive_sessions(self, connection_manager):
        """Test cleaning up inactive sessions."""
        # This would require time-based testing
        # For now, just verify method exists
        await connection_manager.cleanup_inactive_sessions(timeout_seconds=300)

    @pytest.mark.asyncio
    async def test_send_query_start_event(self, connection_manager, mock_websocket, test_api_key):
        """Test sending query start event."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="query_streaming",
        )

        event = WebSocketEvent(
            event="query_start",
            data={
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "What is diabetes?",
                "pipeline": "basic",
            },
            timestamp=datetime.utcnow(),
        )

        await connection_manager.send_to_session(session_id, event)

        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_generation_chunk_event(self, connection_manager, mock_websocket, test_api_key):
        """Test sending generation chunk event."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="query_streaming",
        )

        event = WebSocketEvent(
            event="generation_chunk",
            data={
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "chunk": "Diabetes is ",
                "chunk_index": 0,
            },
            timestamp=datetime.utcnow(),
        )

        await connection_manager.send_to_session(session_id, event)

        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_query_complete_event(self, connection_manager, mock_websocket, test_api_key):
        """Test sending query complete event."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="query_streaming",
        )

        event = WebSocketEvent(
            event="query_complete",
            data={
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "execution_time_ms": 1456,
                "documents_retrieved": 5,
            },
            timestamp=datetime.utcnow(),
        )

        await connection_manager.send_to_session(session_id, event)

        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_upload_progress_event(self, connection_manager, mock_websocket, test_api_key):
        """Test sending document upload progress event."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="document_upload",
        )

        event = WebSocketEvent(
            event="document_upload_progress",
            data={
                "operation_id": "op-123",
                "progress_percentage": 47.5,
                "processed_documents": 47,
                "total_documents": 100,
            },
            timestamp=datetime.utcnow(),
        )

        await connection_manager.send_to_session(session_id, event)

        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_send_error_handling(self, connection_manager, test_api_key):
        """Test error handling when sending message fails."""
        ws = MagicMock()
        ws.send_json = AsyncMock(side_effect=Exception("Connection lost"))
        ws.close = AsyncMock()

        session_id = await connection_manager.connect(ws, test_api_key, "all")

        event = WebSocketEvent(
            event="test",
            data={"message": "test"},
            timestamp=datetime.utcnow(),
        )

        # Should handle error gracefully
        await connection_manager.send_to_session(session_id, event)

        # Session should be cleaned up
        assert len(connection_manager.active_connections) == 0

    @pytest.mark.asyncio
    async def test_filter_by_subscription_type(self, connection_manager, test_api_key):
        """Test filtering connections by subscription type."""
        ws1 = MagicMock()
        ws1.send_json = AsyncMock()
        ws2 = MagicMock()
        ws2.send_json = AsyncMock()

        # Different subscription types
        await connection_manager.connect(ws1, test_api_key, "query_streaming")
        await connection_manager.connect(ws2, test_api_key, "document_upload")

        event = WebSocketEvent(
            event="query_start",
            data={"query": "test"},
            timestamp=datetime.utcnow(),
        )

        # Broadcast only to query_streaming subscribers
        await connection_manager.broadcast_to_subscription(event, "query_streaming")

        ws1.send_json.assert_called_once()
        ws2.send_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnection_token(self, connection_manager, mock_websocket, test_api_key):
        """Test reconnection token generation and validation."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="all",
        )

        token = connection_manager.generate_reconnection_token(session_id)

        assert token is not None
        assert len(token) > 0

        # Validate token
        is_valid = connection_manager.validate_reconnection_token(session_id, token)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_get_session_info(self, connection_manager, mock_websocket, test_api_key):
        """Test retrieving session information."""
        session_id = await connection_manager.connect(
            websocket=mock_websocket,
            api_key=test_api_key,
            subscription_type="all",
        )

        session_info = connection_manager.get_session_info(session_id)

        assert session_info is not None
        assert session_info["session_id"] == session_id
        assert session_info["api_key_id"] == test_api_key.key_id
        assert session_info["subscription_type"] == "all"
