"""
Technique Handler Registry.

Manages technique handlers for the five RAG pipelines the MCP server exposes.
Pipelines are built on first use, so an unimportable pipeline fails its own
tool call and the health check instead of the whole server.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.core.models import Document
from iris_vector_rag.mcp import tool_schemas
from iris_vector_rag.mcp.validation import ValidationError

logger = logging.getLogger(__name__)


def _serialize_document(doc: Any) -> Dict[str, Any]:
    """Convert a Document object to a dictionary for JSON serialization."""
    logger.debug(f"Serializing document type: {type(doc)}, value: {repr(doc)[:200]}")

    if isinstance(doc, Document):
        # Extract core fields from LangChain Document
        serialized = {"content": doc.page_content, **doc.metadata}
        # Add Document.id if it exists (some Documents have this)
        if hasattr(doc, "id") and doc.id:
            serialized["id"] = doc.id
        # Ensure we have either 'id' or 'doc_id' for compatibility
        if "id" not in serialized and "doc_id" not in serialized:
            # Use ticket_id or other ID fields if available
            for id_field in ["ticket_id", "document_id", "source_id"]:
                if id_field in doc.metadata:
                    serialized["doc_id"] = doc.metadata[id_field]
                    break
        # The score travels in metadata as "score" or "similarity"; give it one name.
        if "score" not in serialized and "similarity" in serialized:
            serialized["score"] = serialized["similarity"]
        logger.debug(f"Serialized Document to: {list(serialized.keys())}")
        return serialized
    elif isinstance(doc, dict):
        # Already a dict, return as-is
        logger.debug("Document already dict, returning as-is")
        return doc
    else:
        # Unknown type, try to convert
        logger.warning(f"Unknown document type {type(doc)}, converting to string")
        return {"content": str(doc)}


class TechniqueHandler:
    """Base handler for RAG technique execution."""

    def __init__(self, pipeline_name: str, pipeline_factory):
        self.pipeline_name = pipeline_name
        self.pipeline_factory = pipeline_factory
        self._pipeline_instance = None
        self._last_success = None
        self._error_count = 0
        self._query_count = 0

    # One IRIS session serves every pipeline; queries run one at a time, off the
    # event loop so the stdio transport keeps answering pings and cancellations.
    _run_lock = threading.Lock()

    def _get_pipeline(self):
        """Get or create pipeline instance. Raises if it cannot be built."""
        if self._pipeline_instance is None:
            self._pipeline_instance = self.pipeline_factory()
        return self._pipeline_instance

    def _run_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._run_lock:
            pipeline = self._get_pipeline()
            return pipeline.query(query=query, **params)

    def construct(self) -> None:
        """Build the pipeline now (health check). Raises if it cannot be built."""
        with self._run_lock:
            self._get_pipeline()

    async def execute(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the technique with given query and parameters."""
        try:
            result = await asyncio.to_thread(self._run_query, query, params)

            self._last_success = datetime.now()
            self._query_count += 1

            # Serialize retrieved documents to dict format for MCP/JSON response
            raw_documents = result.get(
                "retrieved_documents", result.get("documents", [])
            )
            serialized_documents = [_serialize_document(doc) for doc in raw_documents]

            # Standardize response format
            # Convert execution_time (seconds) to execution_time_ms (milliseconds)
            execution_time = result.get("execution_time", 0)
            execution_time_ms = int(execution_time * 1000) if execution_time else 0

            return {
                "answer": result.get("answer", ""),
                "retrieved_documents": serialized_documents,
                "sources": result.get("sources", []),
                "metadata": {
                    **result.get("metadata", {}),
                    "pipeline_name": self.pipeline_name,
                },
                "performance": {
                    "execution_time_ms": execution_time_ms,
                    "retrieval_time_ms": result.get("retrieval_time_ms", 0),
                    "generation_time_ms": result.get("generation_time_ms", 0),
                    "tokens_used": result.get("tokens_used", 0),
                },
            }
        except Exception:
            self._error_count += 1
            raise

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against technique's schema."""
        tool_name = f"rag_{self.pipeline_name}"
        return tool_schemas.validate_params(tool_name, params, skip_query=True)

    async def health_check(self) -> Dict[str, Any]:
        """Check health of this technique."""
        error_rate = (
            self._error_count / self._query_count if self._query_count > 0 else 0.0
        )

        if error_rate > 0.5:
            status = "degraded"
        elif self._pipeline_instance is None:
            status = "unavailable"
        else:
            status = "healthy"

        return {
            "status": status,
            "last_success": (
                self._last_success.isoformat() if self._last_success else None
            ),
            "error_rate": error_rate,
        }


class TechniqueHandlerRegistry:
    """
    Registry of all technique handlers.

    Supports strict_mode validation to ensure all registered pipelines
    conform to the RAGPipeline contract (see iris_rag.core.validators).
    """

    def __init__(self, strict_mode: bool = False, validate_on_register: bool = True):
        """
        Initialize the registry.

        Args:
            strict_mode: If True, treat validation warnings as errors
            validate_on_register: If True, validate pipeline classes when registering
        """
        self._handlers: Dict[str, TechniqueHandler] = {}
        self._strict_mode = strict_mode
        self._validate_on_register = validate_on_register
        self._validation_results: Dict[str, List] = {}  # Store validation results
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register the five RAG techniques. Imports happen when a pipeline is built."""

        def basic():
            from iris_vector_rag.pipelines.basic import BasicRAGPipeline

            return BasicRAGPipeline()

        def basic_rerank():
            from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline

            return BasicRAGRerankingPipeline()

        def crag():
            from iris_vector_rag.pipelines.crag import CRAGPipeline

            return CRAGPipeline()

        def graphrag():
            from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

            return HybridGraphRAGPipeline()

        def pylate_colbert():
            from iris_vector_rag.pipelines.colbert_pylate.pylate_pipeline import (
                PyLateColBERTPipeline,
            )

            return PyLateColBERTPipeline()

        for name, factory in (
            ("basic", basic),
            ("basic_rerank", basic_rerank),
            ("crag", crag),
            ("graphrag", graphrag),
            ("pylate_colbert", pylate_colbert),
        ):
            self._handlers[name] = TechniqueHandler(name, factory)

    def register_handler(
        self,
        technique: str,
        handler: TechniqueHandler,
        pipeline_class: Optional[Type[RAGPipeline]] = None,
    ):
        """
        Register a technique handler.

        Args:
            technique: Technique name (e.g., 'basic', 'crag')
            handler: TechniqueHandler instance
            pipeline_class: Optional pipeline class for validation

        Raises:
            ValidationError: If validation fails in strict_mode
        """
        # Validate pipeline class if provided and validation is enabled
        if self._validate_on_register and pipeline_class is not None:
            from iris_vector_rag.core.validators import (
                PipelineValidator,
                ViolationSeverity,
            )

            validator = PipelineValidator(strict_mode=self._strict_mode)
            violations = validator.validate_pipeline_class(pipeline_class)

            # Store validation results
            self._validation_results[technique] = violations

            # Check for errors
            errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
            if errors:
                error_msg = f"Pipeline validation failed for '{technique}':\n"
                error_msg += "\n".join(str(e) for e in errors)
                logger.error(error_msg)
                raise ValidationError(
                    field="pipeline_class", value=technique, message=error_msg
                )

            # In strict_mode, treat warnings as errors
            if self._strict_mode:
                warnings = [
                    v for v in violations if v.severity == ViolationSeverity.WARNING
                ]
                if warnings:
                    error_msg = (
                        f"Pipeline validation failed (strict mode) for '{technique}':\n"
                    )
                    error_msg += "\n".join(str(w) for w in warnings)
                    logger.error(error_msg)
                    raise ValidationError(
                        field="pipeline_class", value=technique, message=error_msg
                    )

            # Log any informational messages
            infos = [v for v in violations if v.severity == ViolationSeverity.INFO]
            for info in infos:
                logger.info(str(info))

        self._handlers[technique] = handler

    def get_handler(self, technique: str) -> TechniqueHandler:
        """Get handler for technique."""
        if technique not in self._handlers:
            raise KeyError(f"Unknown technique: {technique}")
        return self._handlers[technique]

    def list_techniques(self) -> List[str]:
        """List all registered techniques."""
        return list(self._handlers.keys())

    def get_all_handlers(self) -> Dict[str, TechniqueHandler]:
        """Get all registered handlers."""
        return self._handlers.copy()

    def get_validation_results(
        self, technique: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Get validation results for registered techniques.

        Args:
            technique: If provided, return results for specific technique only

        Returns:
            Dictionary mapping technique names to validation violation lists
        """
        if technique:
            return {technique: self._validation_results.get(technique, [])}
        return self._validation_results.copy()

    def validate_all_handlers(self) -> Dict[str, List]:
        """
        Validate all registered handlers against pipeline contract.

        Returns:
            Dictionary mapping technique names to validation violation lists
        """
        from iris_vector_rag.core.validators import PipelineValidator

        validator = PipelineValidator(strict_mode=self._strict_mode)
        results = {}

        for technique, handler in self._handlers.items():
            # Get pipeline instance to validate
            try:
                pipeline = handler._get_pipeline()
                violations = validator.validate_pipeline_class(type(pipeline))
                results[technique] = violations

                # Log violations
                if violations:
                    for violation in violations:
                        logger.warning(f"[{technique}] {violation}")
            except Exception as e:
                logger.error(f"Failed to validate technique '{technique}': {e}")
                results[technique] = []

        self._validation_results = results
        return results
