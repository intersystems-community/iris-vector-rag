"""
Response Standardizer for RAG Pipelines

This module provides response format standardization to ensure all RAG pipelines
return consistent, predictable response formats for integration testing and
client compatibility.

Addresses the critical issue where only 1/7 pipelines returned required keys
(contexts, metadata) causing integration test failures.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union
from .models import Document

logger = logging.getLogger(__name__)


class ResponseStandardizer:
    """
    Standardizes pipeline responses to ensure consistent format across all RAG pipelines.

    This addresses the integration test failure where only BasicRAG returned
    the required keys while 6 other pipelines had inconsistent response formats.
    """

    @staticmethod
    def standardize_response(raw_response: Dict[str, Any], pipeline_type: str = "unknown") -> Dict[str, Any]:
        """
        Transform any pipeline response to standardized format.

        Args:
            raw_response: Raw response from any RAG pipeline
            pipeline_type: Type of pipeline for metadata tracking

        Returns:
            Standardized response with all required keys
        """
        try:
            # Extract required components with robust fallbacks
            query = raw_response.get("query", "")
            answer = raw_response.get("answer", None)

            # Extract and standardize documents
            documents = ResponseStandardizer._extract_documents(raw_response)

            # Generate contexts from documents if missing
            contexts = ResponseStandardizer._generate_contexts(raw_response, documents)

            # Build comprehensive metadata
            metadata = ResponseStandardizer._build_metadata(raw_response, pipeline_type, documents)

            # Normalize timing information
            execution_time = ResponseStandardizer._normalize_timing(raw_response)

            # Create standardized response
            standardized = {
                "query": query,
                "retrieved_documents": documents,
                "contexts": contexts,
                "metadata": metadata,
                "answer": answer,
                "execution_time": execution_time,
            }

            logger.debug(
                f"Standardized response for {pipeline_type}: "
                f"{len(documents)} docs, {len(contexts)} contexts, "
                f"metadata keys: {list(metadata.keys())}"
            )

            return standardized

        except Exception as e:
            logger.error(f"Failed to standardize response for {pipeline_type}: {e}")
            # Return minimal valid response on error
            return {
                "query": raw_response.get("query", ""),
                "retrieved_documents": [],
                "contexts": [],
                "metadata": {
                    "pipeline_type": pipeline_type,
                    "standardization_error": str(e),
                    "original_keys": list(raw_response.keys()),
                },
                "answer": raw_response.get("answer", None),
                "execution_time": 0.0,
            }

    @staticmethod
    def _extract_documents(raw_response: Dict[str, Any]) -> List[Document]:
        """Extract Document objects from various response formats."""
        # Try different common field names
        document_fields = ["retrieved_documents", "documents", "results", "search_results"]

        for field in document_fields:
            if field in raw_response:
                docs = raw_response[field]
                if isinstance(docs, list):
                    # Convert to Document objects if needed
                    return ResponseStandardizer._ensure_document_objects(docs)

        logger.debug("No documents found in response")
        return []

    @staticmethod
    def _ensure_document_objects(docs: List[Any]) -> List[Document]:
        """Ensure all items are Document objects."""
        document_objects = []

        for doc in docs:
            if isinstance(doc, Document):
                document_objects.append(doc)
            elif isinstance(doc, dict):
                # Convert dict to Document object
                doc_obj = Document(
                    id=doc.get("doc_id", doc.get("id", "")),
                    page_content=doc.get("content", doc.get("page_content", "")),
                    metadata=doc.get("metadata", {}),
                )
                document_objects.append(doc_obj)
            else:
                logger.warning(f"Unknown document format: {type(doc)}")
                # Create minimal Document object
                doc_obj = Document(id="unknown", page_content=str(doc), metadata={})
                document_objects.append(doc_obj)

        return document_objects

    @staticmethod
    def _generate_contexts(raw_response: Dict[str, Any], documents: List[Document]) -> List[str]:
        """Generate context strings from documents if not present in response."""
        # Check if contexts already exist
        if "contexts" in raw_response:
            contexts = raw_response["contexts"]
            if isinstance(contexts, list):
                return [str(ctx) for ctx in contexts]

        # Generate contexts from documents
        contexts = []
        for doc in documents:
            if hasattr(doc, "page_content") and doc.page_content:
                contexts.append(str(doc.page_content))
            else:
                contexts.append("")

        logger.debug(f"Generated {len(contexts)} contexts from {len(documents)} documents")
        return contexts

    @staticmethod
    def _build_metadata(raw_response: Dict[str, Any], pipeline_type: str, documents: List[Document]) -> Dict[str, Any]:
        """Build comprehensive metadata from response."""
        metadata = {
            "pipeline_type": pipeline_type,
            "num_retrieved": len(documents),
            "generated_answer": raw_response.get("answer") is not None,
        }

        # Include pipeline-specific metadata
        pipeline_specific_keys = [
            "retrieval_method",  # Various pipelines
            "similarity_scores",  # Vector-based pipelines
            "processing_time",  # Alternative timing field
        ]

        for key in pipeline_specific_keys:
            if key in raw_response:
                metadata[key] = raw_response[key]

        # Include any existing metadata
        if "metadata" in raw_response and isinstance(raw_response["metadata"], dict):
            metadata.update(raw_response["metadata"])

        return metadata

    @staticmethod
    def _normalize_timing(raw_response: Dict[str, Any]) -> float:
        """Extract and normalize timing information."""
        # Try different timing field names
        timing_fields = ["execution_time", "processing_time", "response_time", "query_time", "total_time"]

        for field in timing_fields:
            if field in raw_response:
                timing = raw_response[field]
                if isinstance(timing, (int, float)):
                    return float(timing)

        # No timing found
        return 0.0


def standardize_pipeline_response(response: Dict[str, Any], pipeline_type: str = "unknown") -> Dict[str, Any]:
    """
    Convenience function to standardize pipeline responses.

    Args:
        response: Raw response from any RAG pipeline
        pipeline_type: Type of pipeline for tracking

    Returns:
        Standardized response with all required keys
    """
    return ResponseStandardizer.standardize_response(response, pipeline_type)
