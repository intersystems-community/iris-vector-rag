"""
WebSocket Event Handlers for RAG API.

Implements FR-026 to FR-027: Query streaming and document upload progress.
Provides real-time event streaming for long-running operations.
"""

import logging
import asyncio
from uuid import UUID, uuid4
from typing import AsyncIterator

from iris_vector_rag.api.models.websocket import (
    EventType,
    QueryProgressEvent,
    DocumentUploadProgressEvent
)
from iris_vector_rag.api.websocket.connection import ConnectionManager
from iris_vector_rag.api.services.pipeline_manager import PipelineManager
from iris_vector_rag.api.services.document_service import DocumentService


logger = logging.getLogger(__name__)


class QueryStreamingHandler:
    """
    Handles query streaming over WebSocket.

    Implements FR-026: Stream incremental query results
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        pipeline_manager: PipelineManager
    ):
        """
        Initialize query streaming handler.

        Args:
            connection_manager: WebSocket connection manager
            pipeline_manager: Pipeline manager for query execution
        """
        self.connection_manager = connection_manager
        self.pipeline_manager = pipeline_manager

    async def stream_query(
        self,
        session_id: UUID,
        query: str,
        pipeline_name: str,
        top_k: int,
        request_id: UUID
    ):
        """
        Stream query execution progress.

        Args:
            session_id: WebSocket session ID
            query: Query text
            pipeline_name: Pipeline to use
            top_k: Number of documents to retrieve
            request_id: Request identifier

        Implements FR-026: Stream incremental results
        """
        try:
            # Send query start event
            await self.connection_manager.send_event(
                session_id=session_id,
                event_type=EventType.QUERY_START,
                data=QueryProgressEvent(
                    query=query,
                    pipeline=pipeline_name,
                    is_final=False
                ).model_dump(),
                request_id=request_id
            )

            # Get pipeline
            pipeline = self.pipeline_manager.get_pipeline(pipeline_name)

            if not pipeline:
                # Send error event
                await self.connection_manager.send_event(
                    session_id=session_id,
                    event_type=EventType.ERROR,
                    data={
                        "error": f"Pipeline not found: {pipeline_name}",
                        "is_final": True
                    },
                    request_id=request_id
                )
                return

            # Execute query with streaming (if supported)
            # For now, simulate streaming with progress updates

            # Simulate retrieval progress
            await self.connection_manager.send_event(
                session_id=session_id,
                event_type=EventType.RETRIEVAL_PROGRESS,
                data=QueryProgressEvent(
                    query=query,
                    pipeline=pipeline_name,
                    documents_retrieved=0,
                    is_final=False
                ).model_dump(),
                request_id=request_id
            )

            # Execute actual query
            result = pipeline.query(query=query, top_k=top_k)

            # Send retrieval complete
            await self.connection_manager.send_event(
                session_id=session_id,
                event_type=EventType.RETRIEVAL_PROGRESS,
                data=QueryProgressEvent(
                    query=query,
                    pipeline=pipeline_name,
                    documents_retrieved=len(result.get("retrieved_documents", [])),
                    is_final=False
                ).model_dump(),
                request_id=request_id
            )

            # Stream generation chunks (if LLM supports streaming)
            answer = result.get("answer", "")

            # Simulate chunk streaming (in real implementation, LLM would stream)
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                is_final = (i + chunk_size) >= len(answer)

                await self.connection_manager.send_event(
                    session_id=session_id,
                    event_type=EventType.GENERATION_CHUNK,
                    data=QueryProgressEvent(
                        query=query,
                        pipeline=pipeline_name,
                        documents_retrieved=len(result.get("retrieved_documents", [])),
                        generation_chunk=chunk,
                        is_final=is_final
                    ).model_dump(),
                    request_id=request_id
                )

                # Small delay to simulate streaming
                await asyncio.sleep(0.1)

            # Send query complete
            await self.connection_manager.send_event(
                session_id=session_id,
                event_type=EventType.QUERY_COMPLETE,
                data={
                    "query": query,
                    "pipeline": pipeline_name,
                    "total_documents": len(result.get("retrieved_documents", [])),
                    "execution_time_ms": result.get("metadata", {}).get("execution_time_ms", 0),
                    "is_final": True
                },
                request_id=request_id
            )

            logger.info(
                f"Query streaming completed: session={session_id}, "
                f"pipeline={pipeline_name}"
            )

        except Exception as e:
            logger.error(f"Query streaming failed: {e}", exc_info=True)

            # Send error event
            await self.connection_manager.send_event(
                session_id=session_id,
                event_type=EventType.ERROR,
                data={
                    "error": str(e),
                    "query": query,
                    "pipeline": pipeline_name,
                    "is_final": True
                },
                request_id=request_id
            )


class DocumentUploadProgressHandler:
    """
    Handles document upload progress streaming.

    Implements FR-027: Stream document loading progress
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        document_service: DocumentService
    ):
        """
        Initialize document upload progress handler.

        Args:
            connection_manager: WebSocket connection manager
            document_service: Document service for upload operations
        """
        self.connection_manager = connection_manager
        self.document_service = document_service

    async def stream_upload_progress(
        self,
        session_id: UUID,
        operation_id: UUID,
        request_id: UUID
    ):
        """
        Stream document upload progress.

        Args:
            session_id: WebSocket session ID
            operation_id: Upload operation ID
            request_id: Request identifier

        Implements FR-027: Stream progress with percentage
        """
        try:
            # Poll for progress updates
            last_progress = 0.0

            while True:
                # Get current operation status
                operation = self.document_service.get_operation_status(operation_id)

                if not operation:
                    # Send error event
                    await self.connection_manager.send_event(
                        session_id=session_id,
                        event_type=EventType.ERROR,
                        data={
                            "error": f"Operation not found: {operation_id}",
                            "operation_id": str(operation_id)
                        },
                        request_id=request_id
                    )
                    break

                # Send progress update if changed
                if operation.progress_percentage != last_progress:
                    await self.connection_manager.send_event(
                        session_id=session_id,
                        event_type=EventType.DOCUMENT_UPLOAD_PROGRESS,
                        data=DocumentUploadProgressEvent(
                            operation_id=operation_id,
                            processed_documents=operation.processed_documents,
                            total_documents=operation.total_documents,
                            progress_percentage=operation.progress_percentage
                        ).model_dump(),
                        request_id=request_id
                    )

                    last_progress = operation.progress_percentage

                # Check if operation is complete or failed
                if operation.status.value in ["completed", "failed"]:
                    # Send final event
                    event_data = DocumentUploadProgressEvent(
                        operation_id=operation_id,
                        processed_documents=operation.processed_documents,
                        total_documents=operation.total_documents,
                        progress_percentage=operation.progress_percentage
                    ).model_dump()

                    event_data["status"] = operation.status.value

                    if operation.status.value == "failed":
                        event_data["error_message"] = operation.error_message

                    await self.connection_manager.send_event(
                        session_id=session_id,
                        event_type=EventType.DOCUMENT_UPLOAD_PROGRESS,
                        data=event_data,
                        request_id=request_id
                    )

                    logger.info(
                        f"Upload progress streaming completed: "
                        f"operation={operation_id}, status={operation.status.value}"
                    )
                    break

                # Wait before next poll
                await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Upload progress streaming failed: {e}", exc_info=True)

            # Send error event
            await self.connection_manager.send_event(
                session_id=session_id,
                event_type=EventType.ERROR,
                data={
                    "error": str(e),
                    "operation_id": str(operation_id)
                },
                request_id=request_id
            )

    async def watch_all_uploads(
        self,
        session_id: UUID,
        api_key_id: UUID
    ):
        """
        Watch all upload operations for an API key.

        Args:
            session_id: WebSocket session ID
            api_key_id: API key to watch

        Continuously streams progress for all active uploads.
        """
        logger.info(f"Watching all uploads for API key: {api_key_id}")

        try:
            monitored_operations = set()

            while session_id in self.connection_manager.active_connections:
                # Get all operations for API key
                operations = self.document_service.list_operations(
                    api_key_id=api_key_id,
                    limit=50
                )

                # Find active operations
                for operation in operations:
                    if operation.status.value in ["pending", "validating", "processing"]:
                        # Start monitoring if not already
                        if operation.operation_id not in monitored_operations:
                            monitored_operations.add(operation.operation_id)

                            # Start streaming task for this operation
                            asyncio.create_task(
                                self.stream_upload_progress(
                                    session_id=session_id,
                                    operation_id=operation.operation_id,
                                    request_id=uuid4()
                                )
                            )

                # Wait before next check
                await asyncio.sleep(5.0)

        except Exception as e:
            logger.error(f"Upload watching failed: {e}", exc_info=True)
