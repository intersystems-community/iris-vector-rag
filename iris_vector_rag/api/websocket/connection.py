"""
WebSocket Connection Manager for RAG API.

Implements FR-025 to FR-028: WebSocket streaming with connection management.
Manages active WebSocket sessions with heartbeat and reconnection support.
"""

import logging
import asyncio
import json
from typing import Dict, Optional, Set
from uuid import UUID, uuid4
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from iris_vector_rag.api.models.websocket import (
    WebSocketSession,
    WebSocketEvent,
    WebSocketAuthMessage,
    EventType,
    SubscriptionType
)
from iris_vector_rag.api.models.auth import ApiKey
from iris_vector_rag.api.middleware.auth import ApiKeyAuth


logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages active WebSocket connections.

    Implements FR-025: WebSocket connection lifecycle
    Implements FR-028: JSON event protocol with heartbeat
    """

    def __init__(
        self,
        auth_service: ApiKeyAuth,
        max_connections_per_key: int = 10,
        heartbeat_interval: int = 30,
        idle_timeout: int = 300
    ):
        """
        Initialize connection manager.

        Args:
            auth_service: Authentication service for API key validation
            max_connections_per_key: Maximum concurrent connections per API key
            heartbeat_interval: Heartbeat ping interval in seconds
            idle_timeout: Idle timeout in seconds
        """
        self.auth_service = auth_service
        self.max_connections_per_key = max_connections_per_key
        self.heartbeat_interval = heartbeat_interval
        self.idle_timeout = idle_timeout

        # Active connections: session_id -> (websocket, session)
        self.active_connections: Dict[UUID, tuple[WebSocket, WebSocketSession]] = {}

        # API key -> set of session IDs
        self.connections_by_api_key: Dict[UUID, Set[UUID]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        auth_message: WebSocketAuthMessage
    ) -> WebSocketSession:
        """
        Establish WebSocket connection with authentication.

        Args:
            websocket: WebSocket connection
            auth_message: Authentication message with API key

        Returns:
            WebSocketSession for the connection

        Raises:
            Exception: If authentication fails or connection limit reached

        Implements FR-025: WebSocket authentication
        """
        # Parse API key from base64-encoded credentials
        from iris_vector_rag.api.middleware.auth import ApiKeyAuth

        # Create temporary auth service instance for parsing
        temp_auth = ApiKeyAuth(self.auth_service.connection_pool)
        key_id, key_secret = temp_auth.parse_api_key(auth_message.api_key)

        # Verify API key
        api_key: ApiKey = temp_auth.verify_api_key(key_id, key_secret)

        # Check connection limit per API key
        current_connections = len(
            self.connections_by_api_key.get(api_key.key_id, set())
        )

        if current_connections >= self.max_connections_per_key:
            raise Exception(
                f"Maximum {self.max_connections_per_key} concurrent connections "
                f"per API key exceeded"
            )

        # Create session
        session_id = uuid4()

        session = WebSocketSession(
            session_id=session_id,
            api_key_id=api_key.key_id,
            connected_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow(),
            client_ip="0.0.0.0",  # TODO: Get from request
            subscription_type=auth_message.subscription_type,
            is_active=True,
            message_count=0,
            reconnection_token=f"reconnect_{uuid4()}"
        )

        # Store connection
        self.active_connections[session_id] = (websocket, session)

        # Track by API key
        if api_key.key_id not in self.connections_by_api_key:
            self.connections_by_api_key[api_key.key_id] = set()

        self.connections_by_api_key[api_key.key_id].add(session_id)

        logger.info(
            f"WebSocket connected: session={session_id}, "
            f"api_key={api_key.key_id} ({api_key.name})"
        )

        # Start heartbeat task
        asyncio.create_task(self._heartbeat(session_id))

        return session

    async def disconnect(self, session_id: UUID):
        """
        Disconnect WebSocket session.

        Args:
            session_id: Session to disconnect

        Implements FR-025: Connection cleanup
        """
        if session_id not in self.active_connections:
            return

        websocket, session = self.active_connections[session_id]

        # Mark as inactive
        session.is_active = False

        # Remove from tracking
        del self.active_connections[session_id]

        if session.api_key_id in self.connections_by_api_key:
            self.connections_by_api_key[session.api_key_id].discard(session_id)

            # Clean up empty sets
            if not self.connections_by_api_key[session.api_key_id]:
                del self.connections_by_api_key[session.api_key_id]

        logger.info(
            f"WebSocket disconnected: session={session_id}, "
            f"messages_sent={session.message_count}"
        )

    async def send_event(
        self,
        session_id: UUID,
        event_type: EventType,
        data: dict,
        request_id: UUID
    ):
        """
        Send event to WebSocket client.

        Args:
            session_id: Target session
            event_type: Type of event
            data: Event data
            request_id: Associated request ID

        Implements FR-028: JSON event protocol
        """
        if session_id not in self.active_connections:
            logger.warning(f"Cannot send event - session not found: {session_id}")
            return

        websocket, session = self.active_connections[session_id]

        # Create event
        event = WebSocketEvent(
            event=event_type,
            data=data,
            timestamp=datetime.utcnow(),
            request_id=request_id
        )

        try:
            # Send JSON event
            await websocket.send_json(event.model_dump())

            # Update session
            session.message_count += 1
            session.last_activity_at = datetime.utcnow()

        except (WebSocketDisconnect, ConnectionClosed) as e:
            logger.warning(f"Failed to send event - connection closed: {session_id}")
            await self.disconnect(session_id)

    async def broadcast_event(
        self,
        event_type: EventType,
        data: dict,
        request_id: UUID,
        api_key_id: Optional[UUID] = None,
        subscription_filter: Optional[SubscriptionType] = None
    ):
        """
        Broadcast event to multiple sessions.

        Args:
            event_type: Type of event
            data: Event data
            request_id: Associated request ID
            api_key_id: Optional filter by API key
            subscription_filter: Optional filter by subscription type
        """
        target_sessions = []

        for session_id, (websocket, session) in self.active_connections.items():
            # Filter by API key if specified
            if api_key_id and session.api_key_id != api_key_id:
                continue

            # Filter by subscription type if specified
            if subscription_filter:
                if session.subscription_type != subscription_filter and \
                   session.subscription_type != SubscriptionType.ALL:
                    continue

            target_sessions.append(session_id)

        # Send to all matching sessions
        for session_id in target_sessions:
            await self.send_event(session_id, event_type, data, request_id)

    async def _heartbeat(self, session_id: UUID):
        """
        Send periodic heartbeat pings.

        Args:
            session_id: Session to monitor

        Implements FR-028: 30-second heartbeat
        """
        while session_id in self.active_connections:
            await asyncio.sleep(self.heartbeat_interval)

            if session_id not in self.active_connections:
                break

            websocket, session = self.active_connections[session_id]

            # Check idle timeout
            idle_seconds = (
                datetime.utcnow() - session.last_activity_at
            ).total_seconds()

            if idle_seconds > self.idle_timeout:
                logger.info(f"Session idle timeout: {session_id}")
                await self.disconnect(session_id)
                break

            # Send ping
            try:
                await self.send_event(
                    session_id=session_id,
                    event_type=EventType.PING,
                    data={"timestamp": datetime.utcnow().isoformat()},
                    request_id=uuid4()
                )

            except Exception as e:
                logger.error(f"Heartbeat failed for {session_id}: {e}")
                await self.disconnect(session_id)
                break

    def get_session(self, session_id: UUID) -> Optional[WebSocketSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            WebSocketSession or None if not found
        """
        if session_id in self.active_connections:
            _, session = self.active_connections[session_id]
            return session

        return None

    def get_connection_count(self, api_key_id: Optional[UUID] = None) -> int:
        """
        Get active connection count.

        Args:
            api_key_id: Optional filter by API key

        Returns:
            Number of active connections
        """
        if api_key_id:
            return len(self.connections_by_api_key.get(api_key_id, set()))

        return len(self.active_connections)
