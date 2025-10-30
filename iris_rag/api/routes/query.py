"""
Query API routes for RAG API.

Implements FR-001 to FR-005: Query endpoints with validation and response formatting.
Provides POST /{pipeline}/_search endpoints for all pipelines.
"""

import logging
import time
from uuid import UUID, uuid4
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from iris_rag.api.models.request import QueryRequest
from iris_rag.api.models.response import QueryResponse, Document, DocumentMetadata
from iris_rag.api.models.auth import ApiKey, Permission
from iris_rag.api.models.errors import (
    ErrorResponse,
    ErrorType,
    ErrorInfo,
    ErrorDetails,
    validation_error,
    internal_server_error
)
from iris_rag.api.middleware.auth import ApiKeyAuth
from iris_rag.api.services.pipeline_manager import PipelineManager


logger = logging.getLogger(__name__)


def create_query_router(
    pipeline_manager: PipelineManager,
    auth_service: ApiKeyAuth
) -> APIRouter:
    """
    Create query API router.

    Args:
        pipeline_manager: PipelineManager for query execution
        auth_service: Authentication service

    Returns:
        FastAPI router with query endpoints
    """
    router = APIRouter(prefix="/api/v1", tags=["query"])

    async def require_read_permission(request: Request) -> ApiKey:
        """Dependency to check read permission."""
        return await auth_service(request, required_permission=Permission.READ)

    @router.post(
        "/{pipeline}/_search",
        response_model=QueryResponse,
        responses={
            200: {"description": "Successful query with answer and documents"},
            400: {"description": "Invalid request format"},
            401: {"description": "Authentication required"},
            403: {"description": "Insufficient permissions"},
            422: {"description": "Validation error (query too long, invalid top_k, etc.)"},
            429: {"description": "Rate limit exceeded"},
            500: {"description": "Internal server error"},
            503: {"description": "Pipeline unavailable"}
        },
        summary="Query RAG pipeline",
        description="""
        Execute semantic search query against specified RAG pipeline.

        Returns LLM-generated answer with retrieved documents and sources.
        Response format is 100% compatible with LangChain and RAGAS evaluation frameworks.

        **Rate Limits:**
        - Basic: 60 requests/minute
        - Premium: 100 requests/minute
        - Enterprise: 1000 requests/minute

        **Query Constraints:**
        - Query text: 1-10,000 characters
        - top_k: 1-100 documents

        **Response Format:**
        - `answer`: LLM-generated answer text
        - `retrieved_documents`: List of documents with scores and metadata
        - `contexts`: List of document content (for RAGAS compatibility)
        - `sources`: Source references extracted from metadata
        - `execution_time_ms`: Total query execution time
        """
    )
    async def query_pipeline(
        pipeline: str,
        query_request: QueryRequest,
        request: Request,
        api_key: ApiKey = Depends(require_read_permission)
    ) -> QueryResponse:
        """
        Query RAG pipeline (FR-001).

        Args:
            pipeline: Pipeline name
            query_request: Query parameters
            request: FastAPI request
            api_key: Authenticated API key

        Returns:
            QueryResponse with answer and documents

        Raises:
            HTTPException: On validation or execution errors
        """
        # Get request_id from middleware
        request_id: UUID = getattr(request.state, 'request_id', uuid4())

        start_time = time.time()

        try:
            # Validate query text length (FR-003)
            if len(query_request.query) < 1 or len(query_request.query) > 10000:
                raise HTTPException(
                    status_code=422,
                    detail=validation_error(
                        field="query",
                        rejected_value=f"[{len(query_request.query)} characters]",
                        message="Query must be between 1 and 10000 characters",
                        max_length=10000,
                        min_value=1
                    ).model_dump()
                )

            # Validate top_k (FR-003)
            if query_request.top_k < 1 or query_request.top_k > 100:
                raise HTTPException(
                    status_code=422,
                    detail=validation_error(
                        field="top_k",
                        rejected_value=query_request.top_k,
                        message="top_k must be between 1 and 100",
                        min_value=1,
                        max_value=100
                    ).model_dump()
                )

            # Get pipeline instance (FR-001)
            pipeline_instance = pipeline_manager.get_pipeline(pipeline)

            if not pipeline_instance:
                raise HTTPException(
                    status_code=404,
                    detail=ErrorResponse(
                        error=ErrorInfo(
                            type=ErrorType.BAD_REQUEST,
                            reason=f"Pipeline not found: {pipeline}",
                            details=ErrorDetails(
                                message=f"Pipeline '{pipeline}' does not exist or is not configured"
                            )
                        )
                    ).model_dump()
                )

            # Check pipeline health
            pipeline_metadata = pipeline_manager.get_pipeline_info(pipeline)

            if pipeline_metadata and pipeline_metadata.status.value == "unavailable":
                raise HTTPException(
                    status_code=503,
                    detail=ErrorResponse(
                        error=ErrorInfo(
                            type=ErrorType.SERVICE_UNAVAILABLE,
                            reason="Pipeline is currently unavailable",
                            details=ErrorDetails(
                                pipeline=pipeline,
                                status="unavailable",
                                message=pipeline_metadata.error_message or
                                       "Pipeline is not ready to accept queries"
                            )
                        )
                    ).model_dump()
                )

            logger.info(
                f"Executing query on {pipeline}: {query_request.query[:100]}... "
                f"(api_key={api_key.key_id})"
            )

            # Execute query (FR-001)
            query_start = time.time()

            result = pipeline_instance.query(
                query=query_request.query,
                top_k=query_request.top_k,
                filters=query_request.filters
            )

            query_time_ms = int((time.time() - query_start) * 1000)

            # Parse response (FR-002)
            # Assuming pipeline returns dict with: answer, retrieved_documents, metadata
            retrieved_documents = []

            for doc_data in result.get("retrieved_documents", []):
                # Convert to Document model
                doc = Document(
                    doc_id=UUID(doc_data.get("doc_id", str(uuid4()))),
                    content=doc_data.get("content", ""),
                    score=doc_data.get("score", 0.0),
                    metadata=DocumentMetadata(
                        source=doc_data.get("metadata", {}).get("source", "unknown"),
                        chunk_index=doc_data.get("metadata", {}).get("chunk_index"),
                        page_number=doc_data.get("metadata", {}).get("page_number"),
                        created_at=doc_data.get("metadata", {}).get("created_at")
                    )
                )
                retrieved_documents.append(doc)

            # Extract sources from metadata (FR-002)
            sources = list(set(
                doc.metadata.source for doc in retrieved_documents
            ))

            # Calculate execution times
            total_execution_time_ms = int((time.time() - start_time) * 1000)
            retrieval_time_ms = result.get("metadata", {}).get("retrieval_time_ms")
            generation_time_ms = result.get("metadata", {}).get("generation_time_ms")

            # Create response (FR-002)
            response = QueryResponse(
                response_id=uuid4(),
                request_id=request_id,
                answer=result.get("answer", ""),
                retrieved_documents=retrieved_documents,
                sources=sources,
                pipeline_name=pipeline,
                execution_time_ms=total_execution_time_ms,
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                tokens_used=result.get("metadata", {}).get("tokens_used"),
                confidence_score=result.get("metadata", {}).get("confidence_score"),
                metadata=result.get("metadata")
            )

            # Update pipeline metrics
            pipeline_manager.update_pipeline_metrics(
                pipeline_name=pipeline,
                execution_time_ms=total_execution_time_ms,
                success=True
            )

            logger.info(
                f"Query completed: {pipeline} - {total_execution_time_ms}ms "
                f"({len(retrieved_documents)} docs)"
            )

            return response

        except HTTPException:
            # Re-raise HTTP exceptions (already formatted)
            raise

        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)

            # Update pipeline metrics
            pipeline_manager.update_pipeline_metrics(
                pipeline_name=pipeline,
                execution_time_ms=int((time.time() - start_time) * 1000),
                success=False
            )

            raise HTTPException(
                status_code=500,
                detail=internal_server_error(
                    request_id=str(request_id),
                    message="Query processing failed. Please try again or contact support."
                ).model_dump()
            )

    return router
