"""
Response models for RAG API.

Pydantic models for Query Response entity (Entity 5 from data-model.md).
100% LangChain & RAGAS compatible response format.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, computed_field
from uuid import UUID
from datetime import datetime


class DocumentMetadata(BaseModel):
    """
    Document metadata schema.

    Required fields per data-model.md DocumentMetadata schema.
    """

    source: str = Field(
        ...,
        description="Source filename, URL, or identifier",
        examples=["medical_textbook_ch5.pdf"],
    )

    chunk_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Chunk index for split documents",
    )

    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number in source document",
    )

    created_at: Optional[datetime] = Field(
        default=None,
        description="Document creation timestamp (ISO8601)",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "source": "medical_textbook_ch5.pdf",
                "chunk_index": 3,
                "page_number": 127,
                "created_at": "2025-01-10T00:00:00Z",
            }
        }


class Document(BaseModel):
    """
    Retrieved document schema.

    LangChain-compatible document structure.
    Implements FR-002: Return structured documents with metadata.
    """

    doc_id: UUID = Field(
        ...,
        description="Document identifier",
    )

    content: str = Field(
        ...,
        min_length=1,
        description="Document text content",
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity/relevance score (0.0-1.0)",
    )

    metadata: DocumentMetadata = Field(
        ...,
        description="Document metadata including source",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "doc_id": "1a2b3c4d-5678-90ab-cdef-1234567890ab",
                "content": "Diabetes mellitus is a group of metabolic diseases...",
                "score": 0.95,
                "metadata": {
                    "source": "medical_textbook_ch5.pdf",
                    "chunk_index": 3,
                    "page_number": 127,
                },
            }
        }


class QueryResponse(BaseModel):
    """
    Query Response (Entity 5 from data-model.md).

    Structured result containing generated answer, retrieved documents,
    sources, execution metadata, and optional streaming chunks.

    100% LangChain & RAGAS compatible format.
    Implements FR-002: Return structured responses with answer, documents, sources, metadata.
    """

    response_id: UUID = Field(
        ...,
        description="Unique response identifier",
    )

    request_id: UUID = Field(
        ...,
        description="Associated request identifier (from X-Request-ID header)",
    )

    answer: str = Field(
        ...,
        min_length=1,
        description="LLM-generated answer text",
        examples=[
            "Diabetes is a chronic metabolic disorder characterized by elevated blood glucose levels..."
        ],
    )

    retrieved_documents: List[Document] = Field(
        ...,
        min_length=1,
        description="Retrieved documents with metadata",
    )

    sources: List[str] = Field(
        default_factory=list,
        description="Source references (URLs, filenames, etc.)",
        examples=[["medical_textbook_ch5.pdf"]],
    )

    pipeline_name: str = Field(
        ...,
        description="Pipeline that processed the query",
        examples=["graphrag"],
    )

    execution_time_ms: int = Field(
        ...,
        ge=0,
        description="Total query execution time in milliseconds",
    )

    retrieval_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Time spent retrieving documents",
    )

    generation_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Time spent generating answer",
    )

    tokens_used: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total LLM tokens consumed",
    )

    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Answer confidence score (if available)",
    )

    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pipeline-specific metadata",
    )

    @computed_field
    @property
    def contexts(self) -> List[str]:
        """
        Extract document content for RAGAS evaluation.

        RAGAS compatibility field - returns list of document content strings.
        Implements FR-002: RAGAS-compatible response format.
        """
        return [doc.content for doc in self.retrieved_documents]

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "response_id": "9a8b7c6d-5e4f-3210-9876-543210fedcba",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "answer": "Diabetes is a chronic metabolic disorder characterized by elevated blood glucose levels...",
                "retrieved_documents": [
                    {
                        "doc_id": "1a2b3c4d-5678-90ab-cdef-1234567890ab",
                        "content": "Diabetes mellitus is a group of metabolic diseases...",
                        "score": 0.95,
                        "metadata": {
                            "source": "medical_textbook_ch5.pdf",
                            "chunk_index": 3,
                            "page_number": 127,
                        },
                    }
                ],
                "sources": ["medical_textbook_ch5.pdf"],
                "contexts": [
                    "Diabetes mellitus is a group of metabolic diseases..."
                ],
                "pipeline_name": "graphrag",
                "execution_time_ms": 1456,
                "retrieval_time_ms": 345,
                "generation_time_ms": 1089,
                "tokens_used": 2345,
                "confidence_score": 0.92,
                "metadata": {
                    "graph_entities_found": 5,
                    "graph_relationships_traversed": 12,
                },
            }
        }
