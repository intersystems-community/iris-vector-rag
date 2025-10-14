"""
Hybrid GraphRAG Pipeline with IRIS Graph Core Integration

Enhances the existing GraphRAG pipeline with advanced hybrid search capabilities
from the iris_graph_core module, including RRF fusion and iFind text search.

SECURITY-HARDENED VERSION: No hard-coded credentials, config-driven discovery,
robust error handling, and modular architecture.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..config.manager import ConfigurationManager
from ..core.base import RAGPipeline
from ..core.connection import ConnectionManager
from ..core.exceptions import RAGException
from ..core.models import Document
from ..embeddings.manager import EmbeddingManager
from ..services.entity_extraction import EntityExtractionService
from .graphrag import GraphRAGException, GraphRAGPipeline
from .hybrid_graphrag_discovery import GraphCoreDiscovery
from .hybrid_graphrag_retrieval import HybridRetrievalMethods

logger = logging.getLogger(__name__)


class HybridGraphRAGPipeline(GraphRAGPipeline):
    """
    Enhanced GraphRAG pipeline with hybrid search capabilities.

    Integrates iris_graph_core for:
    - RRF (Reciprocal Rank Fusion) combining vector + text + graph signals
    - HNSW-optimized vector search (50ms performance)
    - Native IRIS iFind text search with stemming/stopwords
    - Multi-modal search fusion

    Security features:
    - No hard-coded credentials or paths
    - Config-driven discovery and connections
    - Graceful fallbacks for missing dependencies
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        vector_store=None,
        schema_manager=None,
    ):
        super().__init__(connection_manager, config_manager, llm_func, vector_store)

        # Store schema manager for iris_graph_core table management
        self.schema_manager = schema_manager

        # Initialize graph core discovery
        self.discovery = GraphCoreDiscovery(config_manager)
        self.retrieval_methods = None

        # Graph core components (will be None if not available)
        self.iris_engine = None
        self.fusion_engine = None
        self.text_engine = None
        self.vector_optimizer = None

        # Initialize graph core integration
        self._initialize_graph_core()

    def _initialize_graph_core(self):
        """Initialize iris_graph_core components - required for HybridGraphRAG."""
        # Import modules - will raise ImportError if not available
        modules = self.discovery.import_graph_core_modules()

        # Get connection configuration
        connection_config = self.discovery.get_connection_config()

        # Create IRIS connection using irisnative to avoid embedded Python issues
        try:
            import irisnative

            logger.info(
                f"Connecting to IRIS at {connection_config['host']}:{connection_config['port']}"
                f"/{connection_config['namespace']} for iris_graph_core"
            )

            iris_connection = irisnative.createConnection(
                connection_config["host"],
                connection_config["port"],
                connection_config["namespace"],
                connection_config["username"],
                connection_config["password"],
            )
        except ImportError as e:
            raise ImportError(
                "Unable to import irisnative module. "
                "Please ensure intersystems-irispython is installed: "
                "pip install intersystems-irispython"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to IRIS at {connection_config['host']}:{connection_config['port']}: {e}"
            ) from e

        # Initialize graph core components
        self.iris_engine = modules["IRISGraphEngine"](iris_connection)
        self.fusion_engine = modules["HybridSearchFusion"](self.iris_engine)
        self.text_engine = modules["TextSearchEngine"](iris_connection)
        self.vector_optimizer = modules["VectorOptimizer"](iris_connection)

        # Initialize retrieval methods
        self.retrieval_methods = HybridRetrievalMethods(
            self.iris_engine,
            self.fusion_engine,
            self.text_engine,
            self.vector_optimizer,
            self.embedding_manager,
        )

        # Check for optimized vector table availability
        self.retrieval_methods.check_hnsw_optimization()

        logger.info("✅ Hybrid GraphRAG pipeline initialized with iris-vector-graph integration")

    def query(
        self,
        query_text: str,
        top_k: int = None,
        method: str = "hybrid",
        generate_answer: bool = True,
        custom_prompt: str = None,
        include_sources: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Enhanced query method with hybrid search capabilities.

        Args:
            query_text: The query string
            top_k: Number of documents to retrieve
            method: Retrieval method - "hybrid", "rrf", "kg", "vector", "text"
            generate_answer: Whether to generate an answer
            custom_prompt: Custom prompt for answer generation
            include_sources: Whether to include source information
            **kwargs: Additional parameters including vector_query, fusion_weights

        Returns:
            Dictionary with query results and metadata
        """
        start_time = time.time()
        start_perf = time.perf_counter()

        if top_k is None:
            top_k = self.default_top_k

        # Validate knowledge graph
        self._validate_knowledge_graph()

        # Route to appropriate retrieval method - iris-vector-graph is required
        if method == "hybrid":
            retrieved_documents, retrieval_method = self._retrieve_via_hybrid_fusion(
                query_text, top_k, **kwargs
            )
        elif method == "rrf":
            retrieved_documents, retrieval_method = self._retrieve_via_rrf(
                query_text, top_k, **kwargs
            )
        elif method == "text":
            retrieved_documents, retrieval_method = self._retrieve_via_enhanced_text(
                query_text, top_k, **kwargs
            )
        elif method == "vector":
            retrieved_documents, retrieval_method = self._retrieve_via_hnsw_vector(
                query_text, top_k, **kwargs
            )
        elif method == "kg":
            # Use parent class knowledge graph traversal with fallback
            try:
                retrieved_documents, retrieval_method = self._retrieve_via_kg(
                    query_text, top_k
                )
            except GraphRAGException as e:
                logger.warning(f"GraphRAG fallback: {e}")
                # Fall back to vector search when entities/graph traversal fails
                fallback_messages = [
                    "No seed entities found",
                    "No documents found",
                    "Graph traversal found no additional entities",
                    "Knowledge graph may lack relationships"
                ]
                if any(msg in str(e) for msg in fallback_messages):
                    logger.info(
                        f"GraphRAG: Falling back to vector search for query: '{query_text}'"
                    )
                    retrieved_documents = self._fallback_to_vector_search(query_text, top_k)
                    retrieval_method = "vector_fallback"
                else:
                    # Re-raise other GraphRAG exceptions (validation failures, etc.)
                    raise
        else:
            raise ValueError(
                f"Invalid retrieval method: {method}. "
                f"Supported methods: hybrid, rrf, text, vector, kg"
            )

        # Generate answer if requested
        answer = None
        if generate_answer and self.llm_func and retrieved_documents:
            try:
                answer = self._generate_answer(
                    query_text, retrieved_documents, custom_prompt
                )
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                answer = "Error generating answer"
        elif not generate_answer:
            answer = None
        elif not retrieved_documents:
            answer = "No relevant documents found to answer the query."
        else:
            answer = "No LLM function provided. Retrieved documents only."

        execution_time = time.time() - start_time
        execution_time_ms = (time.perf_counter() - start_perf) * 1000.0

        response = {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents,
            "contexts": [doc.page_content for doc in retrieved_documents],
            "execution_time": execution_time,
            "metadata": {
                "num_retrieved": len(retrieved_documents),
                "processing_time": execution_time,
                "processing_time_ms": execution_time_ms,
                "pipeline_type": "hybrid_graphrag",
                "retrieval_method": retrieval_method,
                "generated_answer": generate_answer and answer is not None,
                "iris_graph_core_enabled": True,
            },
        }

        if include_sources:
            response["sources"] = self._extract_sources(retrieved_documents)

        logger.info(
            f"Hybrid GraphRAG query completed in {execution_time:.2f}s ({execution_time_ms:.1f}ms) - "
            f"{len(retrieved_documents)} docs via {retrieval_method}"
        )
        return response

    def _retrieve_via_hybrid_fusion(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Retrieve documents using multi-modal hybrid search fusion."""
        documents, method = self.retrieval_methods.retrieve_via_hybrid_fusion(
            query_text, top_k, self._get_document_content_for_entity, **kwargs
        )
        if not documents:
            logger.warning("Hybrid fusion returned 0 results")
        return documents, method

    def _retrieve_via_rrf(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Retrieve documents using RRF (Reciprocal Rank Fusion)."""
        documents, method = self.retrieval_methods.retrieve_via_rrf(
            query_text, top_k, self._get_document_content_for_entity, **kwargs
        )
        if not documents:
            logger.warning("RRF returned 0 results")
        return documents, method

    def _retrieve_via_enhanced_text(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Retrieve documents using enhanced IRIS iFind text search."""
        documents, method = self.retrieval_methods.retrieve_via_enhanced_text(
            query_text, top_k, self._get_document_content_for_entity, **kwargs
        )
        if not documents:
            logger.warning("Text search returned 0 results")
        return documents, method

    def _retrieve_via_hnsw_vector(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Retrieve documents using HNSW-optimized vector search."""
        documents, method = self.retrieval_methods.retrieve_via_hnsw_vector(
            query_text, top_k, self._get_document_content_for_entity, **kwargs
        )
        if not documents:
            logger.warning("HNSW vector search returned 0 results")
        return documents, method


    def _get_document_content_for_entity(self, entity_id: str) -> Optional[str]:
        """
        Get document content associated with an entity ID.

        Robust implementation with proper cursor cleanup and error handling.
        """
        connection = None
        cursor = None
        try:
            connection = self.connection_manager.get_connection()
            cursor = connection.cursor()

            # Try to find document content via entity relationships
            cursor.execute(
                """
                SELECT s.text_content
                FROM RAG.SourceDocuments s
                JOIN RAG.Entities e ON e.source_doc_id = s.doc_id
                WHERE e.entity_id = ?
                LIMIT 1
            """,
                [entity_id],
            )

            result = cursor.fetchone()
            return result[0] if result else f"Entity: {entity_id}"

        except Exception as e:
            logger.warning(
                f"Could not get document content for entity {entity_id}: {e}"
            )
            return f"Entity: {entity_id}"
        finally:
            # Robust cursor cleanup - cursor is initialized to None, so this is safe
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error closing cursor: {e}")

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for hybrid search components"""
        return self.retrieval_methods.get_performance_statistics(self.connection_manager)

    def benchmark_search_methods(
        self, query_text: str, iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark different search methods for performance comparison"""
        return self.retrieval_methods.benchmark_search_methods(
            query_text, self.query, iterations
        )

    def is_hybrid_enabled(self) -> bool:
        """Check if hybrid capabilities are enabled - always True for HybridGraphRAG."""
        return True

    def get_hybrid_status(self) -> Dict[str, Any]:
        """Get detailed status of hybrid capabilities."""
        return {
            "hybrid_enabled": True,
            "iris_engine_available": True,
            "fusion_engine_available": True,
            "text_engine_available": True,
            "vector_optimizer_available": True,
            "iris_vector_graph_installed": True,
        }
