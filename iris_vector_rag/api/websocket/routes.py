"""
WebSocket API routes for RAG API.

Implements FR-025 to FR-028: WebSocket endpoints for real-time streaming.
Provides /ws endpoint for query and document upload streaming.
"""

import logging
import json
from uuid import UUID, uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from iris_vector_rag.api.models.websocket import (
    WebSocketAuthMessage,
    EventType,
    SubscriptionType
)
from iris_vector_rag.api.websocket.connection import ConnectionManager
from iris_vector_rag.api.websocket.handlers import (
    QueryStreamingHandler,
    DocumentUploadProgressHandler
)


logger = logging.getLogger(__name__)


def create_websocket_router(
    connection_manager: ConnectionManager,
    query_handler: QueryStreamingHandler,
    upload_handler: DocumentUploadProgressHandler
) -> APIRouter:
    """
    Create WebSocket API router.

    Args:
        connection_manager: WebSocket connection manager
        query_handler: Query streaming handler
        upload_handler: Upload progress handler

    Returns:
        FastAPI router with WebSocket endpoint
    """
    router = APIRouter(tags=["websocket"])

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time streaming.

        Implements FR-025 to FR-028: Full WebSocket support

        **Authentication:**
        First message must be authentication with format:
        ```json
        {
          "api_key": "base64-encoded-key",
          "subscription_type": "query_streaming" | "document_upload" | "all"
        }
        ```

        **Event Protocol:**
        All events follow JSON format:
        ```json
        {
          "event": "event_type",
          "data": {...},
          "timestamp": "ISO8601",
          "request_id": "uuid"
        }
        ```

        **Event Types:**
        - query_start: Query execution started
        - retrieval_progress: Documents being retrieved
        - generation_chunk: Partial answer text
        - query_complete: Query finished
        - document_upload_progress: Upload progress update
        - error: Error occurred
        - ping/pong: Heartbeat

        **Usage Example:**
        ```python
        import websockets
        import json
        import base64

        async with websockets.connect("ws://localhost:8000/ws") as ws:
            # Authenticate
            auth = {
                "api_key": base64.b64encode(b"key-id:secret").decode(),
                "subscription_type": "all"
            }
            await ws.send(json.dumps(auth))

            # Receive events
            async for message in ws:
                event = json.loads(message)
                print(f"Event: {event['event']}, Data: {event['data']}")
        ```
        """
        session_id = None

        try:
            # Accept connection
            await websocket.accept()

            # Wait for authentication message
            auth_data = await websocket.receive_json()

            # Parse authentication
            auth_message = WebSocketAuthMessage(**auth_data)

            # Connect with authentication
            session = await connection_manager.connect(websocket, auth_message)
            session_id = session.session_id

            logger.info(
                f"WebSocket authenticated: session={session_id}, "
                f"subscription={session.subscription_type.value}"
            )

            # Start upload watching if subscribed
            if session.subscription_type in [
                SubscriptionType.DOCUMENT_UPLOAD,
                SubscriptionType.ALL
            ]:
                # Start watching all uploads for this API key
                # (non-blocking background task)
                import asyncio
                asyncio.create_task(
                    upload_handler.watch_all_uploads(
                        session_id=session_id,
                        api_key_id=session.api_key_id
                    )
                )

            # Main message loop
            while True:
                try:
                    # Receive message from client
                    message = await websocket.receive_json()

                    # Handle different message types
                    message_type = message.get("type")

                    if message_type == "query":
                        # Client requesting query streaming
                        if session.subscription_type not in [
                            SubscriptionType.QUERY_STREAMING,
                            SubscriptionType.ALL
                        ]:
                            await connection_manager.send_event(
                                session_id=session_id,
                                event_type=EventType.ERROR,
                                data={
                                    "error": "Not subscribed to query_streaming events"
                                },
                                request_id=uuid4()
                            )
                            continue

                        # Start query streaming (non-blocking)
                        import asyncio
                        asyncio.create_task(
                            query_handler.stream_query(
                                session_id=session_id,
                                query=message.get("query"),
                                pipeline_name=message.get("pipeline", "basic"),
                                top_k=message.get("top_k", 5),
                                request_id=UUID(message.get("request_id", str(uuid4())))
                            )
                        )

                    elif message_type == "watch_upload":
                        # Client requesting upload progress
                        if session.subscription_type not in [
                            SubscriptionType.DOCUMENT_UPLOAD,
                            SubscriptionType.ALL
                        ]:
                            await connection_manager.send_event(
                                session_id=session_id,
                                event_type=EventType.ERROR,
                                data={
                                    "error": "Not subscribed to document_upload events"
                                },
                                request_id=uuid4()
                            )
                            continue

                        # Start upload progress streaming (non-blocking)
                        import asyncio
                        asyncio.create_task(
                            upload_handler.stream_upload_progress(
                                session_id=session_id,
                                operation_id=UUID(message.get("operation_id")),
                                request_id=UUID(message.get("request_id", str(uuid4())))
                            )
                        )

                    elif message_type == "pong":
                        # Client responding to ping
                        logger.debug(f"Pong received from session: {session_id}")

                    else:
                        # Unknown message type
                        await connection_manager.send_event(
                            session_id=session_id,
                            event_type=EventType.ERROR,
                            data={
                                "error": f"Unknown message type: {message_type}"
                            },
                            request_id=uuid4()
                        )

                except WebSocketDisconnect:
                    logger.info(f"Client disconnected: session={session_id}")
                    break

                except Exception as e:
                    logger.error(f"Message handling error: {e}", exc_info=True)

                    await connection_manager.send_event(
                        session_id=session_id,
                        event_type=EventType.ERROR,
                        data={"error": str(e)},
                        request_id=uuid4()
                    )

        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)

            # Try to send error if possible
            if session_id:
                try:
                    await connection_manager.send_event(
                        session_id=session_id,
                        event_type=EventType.ERROR,
                        data={"error": str(e)},
                        request_id=uuid4()
                    )
                except:
                    pass

        finally:
            # Cleanup connection
            if session_id:
                await connection_manager.disconnect(session_id)

    return router
