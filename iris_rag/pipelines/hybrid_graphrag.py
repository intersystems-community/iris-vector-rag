"""
Hybrid GraphRAG Pipeline with IRIS Graph Core Integration

Enhances the existing GraphRAG pipeline with advanced hybrid search capabilities
from the iris_vector_graph module, including RRF fusion and iFind text search.

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

    Integrates iris_vector_graph for:
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
        embedding_config: Optional[str] = None,
    ):
        super().__init__(connection_manager, config_manager, llm_func, vector_store)

        # Store schema manager for iris_vector_graph table management
        self.schema_manager = schema_manager

        # IRIS EMBEDDING configuration (Feature 051)
        self.embedding_config = embedding_config
        self.use_iris_embedding = embedding_config is not None

        if self.use_iris_embedding:
            logger.info(
                f"HybridGraphRAGPipeline initialized with IRIS EMBEDDING auto-vectorization "
                f"and entity extraction (config: {self.embedding_config})"
            )

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
        """Initialize iris_vector_graph components with secure configuration."""
        success, modules = self.discovery.import_graph_core_modules()

        if not success:
            logger.warning(
                "iris_vector_graph not available - using standard GraphRAG only"
            )
            return

        try:
            # Get secure connection configuration
            connection_config = self.discovery.get_connection_config()
            is_valid, missing_params = self.discovery.validate_connection_config(
                connection_config
            )

            if not is_valid:
                logger.warning(
                    f"IRIS connection parameters missing: {missing_params}. "
                    "Disabling iris_vector_graph integration."
                )
                return

            # Create IRIS connection using validated config
            import iris

            logger.info(
                f"Connecting to IRIS at {connection_config['host']}:{connection_config['port']}"
                f"/{connection_config['namespace']} for iris_vector_graph"
            )

            iris_connection = iris.connect(
                connection_config["host"],
                connection_config["port"],
                connection_config["namespace"],
                connection_config["username"],
                connection_config["password"],
            )

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

            logger.info(
                f"✅ Hybrid GraphRAG pipeline initialized with {modules['package_name']} integration"
            )

        except Exception as e:
            logger.error(f"Failed to initialize iris_vector_graph components: {e}")
            logger.warning("Falling back to standard GraphRAG implementation")
            self.iris_engine = None
            self.retrieval_methods = None

    def query(
        self,
        query: str = None,
        query_text: str = None,
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
            query: The query string (standard parameter name)
            query_text: Alias for query (deprecated, for backward compatibility)
            top_k: Number of documents to retrieve
            method: Retrieval method - "hybrid", "rrf", "kg", "vector", "text"
            generate_answer: Whether to generate an answer
            custom_prompt: Custom prompt for answer generation
            include_sources: Whether to include source information
            **kwargs: Additional parameters including vector_query, fusion_weights

        Returns:
            Dictionary with query results and metadata
        """
        # Support both 'query' (standard) and 'query_text' (backward compatibility)
        if query is None and query_text is None:
            raise ValueError("Either 'query' or 'query_text' parameter is required")
        query_text = query if query is not None else query_text

        start_time = time.time()
        start_perf = time.perf_counter()

        if top_k is None:
            top_k = self.default_top_k

        # Validate knowledge graph
        self._validate_knowledge_graph()

        # Route to appropriate retrieval method
        if method == "hybrid" and self.retrieval_methods:
            retrieved_documents, retrieval_method = self._retrieve_via_hybrid_fusion(
                query_text, top_k, **kwargs
            )
        elif method == "rrf" and self.retrieval_methods:
            retrieved_documents, retrieval_method = self._retrieve_via_rrf(
                query_text, top_k, **kwargs
            )
        elif method == "text" and self.retrieval_methods:
            retrieved_documents, retrieval_method = self._retrieve_via_enhanced_text(
                query_text, top_k, **kwargs
            )
        elif method == "vector" and self.retrieval_methods:
            retrieved_documents, retrieval_method = self._retrieve_via_hnsw_vector(
                query_text, top_k, **kwargs
            )
        else:
            # Enhanced fallback with intelligent hybrid search strategies
            logger.info(f"Using enhanced fallback hybrid search for method: {method}")
            retrieved_documents, retrieval_method = self._enhanced_hybrid_fallback(
                query_text, top_k, method, **kwargs
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
                "iris_vector_graph_enabled": self.iris_engine is not None,
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
        try:
            documents, method = self.retrieval_methods.retrieve_via_hybrid_fusion(
                query_text, top_k, self._get_document_content_for_entity, **kwargs
            )
            # If iris_vector_graph returns 0 results, fall back to vector search
            if not documents:
                logger.warning(
                    "Hybrid fusion returned 0 results. Falling back to vector search."
                )
                fallback_docs = self._fallback_to_vector_search(query_text, top_k)
                return fallback_docs, "vector_fallback"
            return documents, method
        except Exception as e:
            logger.error(f"Hybrid fusion retrieval failed: {e}")
            # Fallback to vector search instead of KG traversal
            logger.info("Falling back to IRISVectorStore vector search")
            fallback_docs = self._fallback_to_vector_search(query_text, top_k)
            return fallback_docs, "vector_fallback"

    def _retrieve_via_rrf(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Retrieve documents using RRF (Reciprocal Rank Fusion)."""
        try:
            documents, method = self.retrieval_methods.retrieve_via_rrf(
                query_text, top_k, self._get_document_content_for_entity, **kwargs
            )
            # If RRF returns 0 results, fall back to vector search
            if not documents:
                logger.warning("RRF returned 0 results. Falling back to vector search.")
                fallback_docs = self._fallback_to_vector_search(query_text, top_k)
                return fallback_docs, "vector_fallback"
            return documents, method
        except Exception as e:
            logger.error(f"RRF retrieval failed: {e}")
            logger.info("Falling back to IRISVectorStore vector search")
            fallback_docs = self._fallback_to_vector_search(query_text, top_k)
            return fallback_docs, "vector_fallback"

    def _retrieve_via_enhanced_text(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Retrieve documents using enhanced IRIS iFind text search."""
        try:
            documents, method = self.retrieval_methods.retrieve_via_enhanced_text(
                query_text, top_k, self._get_document_content_for_entity, **kwargs
            )
            # If text search returns 0 results, fall back to vector search
            if not documents:
                logger.warning("Text search returned 0 results. Falling back to vector search.")
                fallback_docs = self._fallback_to_vector_search(query_text, top_k)
                return fallback_docs, "vector_fallback"
            return documents, method
        except Exception as e:
            logger.error(f"Enhanced text retrieval failed: {e}")
            logger.info("Falling back to IRISVectorStore vector search")
            fallback_docs = self._fallback_to_vector_search(query_text, top_k)
            return fallback_docs, "vector_fallback"

    def _retrieve_via_hnsw_vector(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Retrieve documents using HNSW-optimized vector search."""
        try:
            documents, method = self.retrieval_methods.retrieve_via_hnsw_vector(
                query_text, top_k, self._get_document_content_for_entity, **kwargs
            )
            # If HNSW returns 0 results, fall back to IRISVectorStore
            if not documents:
                logger.warning("HNSW vector search returned 0 results. Falling back to IRISVectorStore.")
                fallback_docs = self._fallback_to_vector_search(query_text, top_k)
                return fallback_docs, "vector_fallback"
            return documents, method
        except Exception as e:
            logger.error(f"HNSW vector retrieval failed: {e}")
            logger.info("Falling back to IRISVectorStore vector search")
            fallback_docs = self._fallback_to_vector_search(query_text, top_k)
            return fallback_docs, "vector_fallback"

    def _enhanced_hybrid_fallback(
        self, query_text: str, top_k: int, method: str, **kwargs
    ) -> tuple[List[Document], str]:
        """
        Enhanced hybrid search fallback when iris_vector_graph is not available.

        Implements intelligent search strategies combining knowledge graph traversal
        with vector search to provide hybrid capabilities.
        """
        try:
            # Analyze query to determine best hybrid strategy
            query_analysis = self._analyze_query_for_hybrid_strategy(query_text)

            if method == "hybrid" or method == "rrf":
                # Intelligent hybrid: try KG first, then combine with vector search
                return self._intelligent_hybrid_search(
                    query_text, top_k, query_analysis, **kwargs
                )

            elif method == "text":
                # Enhanced text search using entity matching + vector fallback
                return self._enhanced_text_search_fallback(query_text, top_k, **kwargs)

            elif method == "vector":
                # Enhanced vector search with entity context
                return self._enhanced_vector_search_fallback(
                    query_text, top_k, **kwargs
                )

            else:
                # Default to enhanced knowledge graph traversal
                return self._retrieve_via_kg(query_text, top_k)

        except Exception as e:
            logger.error(f"Enhanced hybrid fallback failed: {e}")
            # Final fallback to standard GraphRAG
            return super()._retrieve_via_kg(query_text, top_k)

    def _analyze_query_for_hybrid_strategy(self, query_text: str) -> Dict[str, Any]:
        """Analyze query to determine the best hybrid search strategy."""
        analysis = {
            "has_medical_entities": False,
            "entity_density": 0,
            "query_type": "general",
            "recommended_strategy": "kg_first",
        }

        # Check for medical entities in query
        medical_keywords = [
            "symptom",
            "disease",
            "treatment",
            "drug",
            "medication",
            "therapy",
            "covid",
            "diabetes",
            "cancer",
            "heart",
            "blood",
            "pain",
            "fever",
        ]

        query_lower = query_text.lower()
        medical_matches = sum(
            1 for keyword in medical_keywords if keyword in query_lower
        )

        analysis["has_medical_entities"] = medical_matches > 0
        analysis["entity_density"] = medical_matches / len(query_text.split())

        # Determine query type
        if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            analysis["query_type"] = "factual"
        elif any(
            word in query_lower for word in ["treat", "cure", "prevent", "manage"]
        ):
            analysis["query_type"] = "treatment"
        elif any(word in query_lower for word in ["symptom", "sign", "cause"]):
            analysis["query_type"] = "diagnostic"

        # Recommend strategy based on analysis
        if analysis["entity_density"] > 0.3:
            analysis["recommended_strategy"] = "kg_primary"
        elif analysis["has_medical_entities"]:
            analysis["recommended_strategy"] = "kg_vector_hybrid"
        else:
            analysis["recommended_strategy"] = "vector_primary"

        return analysis

    def _intelligent_hybrid_search(
        self, query_text: str, top_k: int, analysis: Dict[str, Any], **kwargs
    ) -> tuple[List[Document], str]:
        """Intelligent hybrid search combining KG and vector strategies."""
        strategy = analysis["recommended_strategy"]

        if strategy == "kg_primary":
            # Try KG first, supplement with vector if needed
            try:
                kg_docs, kg_method = self._retrieve_via_kg(query_text, top_k)
                if len(kg_docs) >= top_k * 0.7:  # Good KG coverage
                    return kg_docs, f"{kg_method}_primary"
                else:
                    # Supplement with vector search
                    vector_docs = self._fallback_to_vector_search(
                        query_text, top_k - len(kg_docs)
                    )
                    combined_docs = kg_docs + vector_docs[: top_k - len(kg_docs)]
                    return combined_docs, "kg_vector_hybrid"
            except Exception:
                return (
                    self._fallback_to_vector_search(query_text, top_k),
                    "vector_fallback",
                )

        elif strategy == "kg_vector_hybrid":
            # Balance between KG and vector
            try:
                kg_docs, _ = self._retrieve_via_kg(query_text, max(2, top_k // 2))
                vector_docs = self._fallback_to_vector_search(query_text, top_k)

                # Merge and deduplicate
                combined_docs = self._merge_and_rank_results(
                    kg_docs, vector_docs, top_k
                )
                return combined_docs, "balanced_hybrid"
            except Exception:
                return (
                    self._fallback_to_vector_search(query_text, top_k),
                    "vector_fallback",
                )

        else:  # vector_primary
            # Vector first with KG enhancement
            vector_docs = self._fallback_to_vector_search(query_text, top_k)
            return vector_docs, "vector_primary"

    def _enhanced_text_search_fallback(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Enhanced text search using entity matching and content filtering."""
        try:
            # First try to find relevant entities
            seed_entities = self._find_seed_entities(query_text)

            if seed_entities:
                # Use entity-guided document retrieval
                entity_docs, _ = self._get_documents_from_entities(
                    {e[0] for e in seed_entities}, top_k
                )
                return entity_docs, "entity_guided_text"
            else:
                # Fallback to vector search
                return (
                    self._fallback_to_vector_search(query_text, top_k),
                    "text_vector_fallback",
                )

        except Exception as e:
            logger.warning(f"Enhanced text search fallback failed: {e}")
            vector_docs = self._fallback_to_vector_search(query_text, top_k)
            return vector_docs, "vector_fallback"

    def _enhanced_vector_search_fallback(
        self, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """Enhanced vector search with knowledge graph context."""
        try:
            # Get base vector results
            vector_docs = self._fallback_to_vector_search(query_text, top_k * 2)

            # Try to enhance with entity context
            try:
                seed_entities = self._find_seed_entities(query_text)
                if seed_entities:
                    entity_docs, _ = self._get_documents_from_entities(
                        {e[0] for e in seed_entities}, top_k
                    )
                    # Merge entity context with vector results
                    enhanced_docs = self._merge_and_rank_results(
                        entity_docs, vector_docs, top_k
                    )
                    return enhanced_docs, "vector_entity_enhanced"
            except Exception:
                pass

            return vector_docs[:top_k], "vector_only"

        except Exception as e:
            logger.error(f"Enhanced vector search failed: {e}")
            return [], "vector_search_failed"

    def _merge_and_rank_results(
        self, primary_docs: List[Document], secondary_docs: List[Document], top_k: int
    ) -> List[Document]:
        """Merge and rank results from different search methods."""
        # Simple deduplication by content hash
        seen_content = set()
        merged_docs = []

        # Add primary docs first (higher priority)
        for doc in primary_docs:
            content_hash = hash(doc.page_content[:200])  # Hash first 200 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                merged_docs.append(doc)
                if len(merged_docs) >= top_k:
                    break

        # Add secondary docs if we need more
        for doc in secondary_docs:
            if len(merged_docs) >= top_k:
                break
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                merged_docs.append(doc)

        return merged_docs[:top_k]

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
                JOIN RAG.Entities e ON e.source_doc_id = s.id
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
        if self.retrieval_methods:
            return self.retrieval_methods.get_performance_statistics(
                self.connection_manager
            )
        else:
            return {"iris_vector_graph_enabled": False}

    def benchmark_search_methods(
        self, query_text: str, iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark different search methods for performance comparison"""
        if self.retrieval_methods:
            return self.retrieval_methods.benchmark_search_methods(
                query_text, self.query, iterations
            )
        else:
            logger.warning("iris_vector_graph not available for benchmarking")
            return {}

    def is_hybrid_enabled(self) -> bool:
        """Check if hybrid capabilities are enabled."""
        return self.iris_engine is not None and self.retrieval_methods is not None

    def get_hybrid_status(self) -> Dict[str, Any]:
        """Get detailed status of hybrid capabilities."""
        return {
            "hybrid_enabled": self.is_hybrid_enabled(),
            "iris_engine_available": self.iris_engine is not None,
            "fusion_engine_available": self.fusion_engine is not None,
            "text_engine_available": self.text_engine is not None,
            "vector_optimizer_available": self.vector_optimizer is not None,
            "graph_core_path": (
                str(self.discovery.discover_graph_core_path())
                if self.discovery.discover_graph_core_path()
                else None
            ),
        }
