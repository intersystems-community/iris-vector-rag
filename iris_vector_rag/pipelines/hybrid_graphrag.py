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
from ..core.connection import ConnectionManager
from ..core.models import Document
from .graphrag import GraphRAGPipeline
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
        executor=None,
    ):
        super().__init__(
            connection_manager,
            config_manager,
            llm_func,
            vector_store,
            executor=executor,
        )

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
        try:
            modules = self.discovery.import_graph_core_modules()
            # Get secure connection configuration
            if hasattr(self, "config_manager") and self.config_manager is not None:
                connection_config = self.config_manager.get_database_config()
            else:
                connection_config = self.discovery.get_connection_config()
            is_valid, missing_params = self.discovery.validate_connection_config(
                connection_config
            )

            if not is_valid:
                raise ValueError(
                    f"IRIS connection parameters missing: {missing_params}"
                )

            # Create IRIS connection using validated config
            import iris

            logger.info(
                f"Connecting to IRIS at {connection_config['host']}:{connection_config['port']}"
                f"/{connection_config['namespace']} for iris_vector_graph"
            )

            host = connection_config["host"]
            port = connection_config["port"]
            namespace = connection_config["namespace"]
            username = connection_config["username"]
            password = connection_config["password"]

            if None in (host, port, namespace, username, password):
                raise ValueError("IRIS connection parameters are incomplete")

            iris_connection = iris.createConnection(  # type: ignore[attr-defined]
                str(host),
                int(port),
                str(namespace),
                str(username),
                str(password),
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
                "✅ Hybrid GraphRAG pipeline initialized with iris_vector_graph integration"
            )

        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize iris_vector_graph components: {e}")
            raise

    def query(
        self,
        query_text: Optional[str] = None,
        top_k: int = 10,
        generate_answer: bool = True,
        query: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Enhanced query method with hybrid search capabilities.

        Args:
            query_text: Query string
            top_k: Number of documents to retrieve
            generate_answer: Whether to generate an answer
            **kwargs: Additional parameters including method, custom_prompt, include_sources

        Returns:
            Dictionary with query results and metadata
        """
        if query is not None:
            query_text = query

        method = kwargs.pop("method", "hybrid")
        custom_prompt = kwargs.pop("custom_prompt", None)
        include_sources = kwargs.pop("include_sources", False)
        query_override = kwargs.pop("query", None)
        if query_override is not None:
            query_text = query_override

        if not query_text:
            raise ValueError("Query text cannot be empty")

        start_time = time.time()
        start_perf = time.perf_counter()

        if top_k is None:
            top_k = self.default_top_k

        retrieval_methods = self.retrieval_methods

        # Validate knowledge graph
        self._validate_knowledge_graph()

        # Route to appropriate retrieval method
        if method == "hybrid":
            if retrieval_methods is None:
                raise RuntimeError("Hybrid retrieval methods not initialized")
            retrieved_documents, retrieval_method = self._retrieve_via_hybrid_fusion(
                query_text, top_k, **kwargs
            )
        elif method == "rrf":
            if retrieval_methods is None:
                raise RuntimeError("Hybrid retrieval methods not initialized")
            retrieved_documents, retrieval_method = self._retrieve_via_rrf(
                query_text, top_k, **kwargs
            )
        elif method == "text":
            if retrieval_methods is None:
                raise RuntimeError("Hybrid retrieval methods not initialized")
            retrieved_documents, retrieval_method = self._retrieve_via_enhanced_text(
                query_text, top_k, **kwargs
            )
        elif method == "vector":
            if retrieval_methods is None:
                raise RuntimeError("Hybrid retrieval methods not initialized")
            retrieved_documents, retrieval_method = self._retrieve_via_hnsw_vector(
                query_text, top_k, **kwargs
            )
        else:
            logger.info(f"Using vector fallback search for method: {method}")
            retrieved_documents = self._fallback_to_vector_search(query_text, top_k)
            retrieval_method = "vector_fallback"

        if method == "kg" and retrieval_method == "no_matching_entities":
            logger.info(
                "No matching entities for KG traversal; reporting vector_fallback"
            )
            retrieval_method = "vector_fallback"

        # Normalize documents for similarity score expectations
        retrieved_documents = self._ensure_similarity_scores(retrieved_documents)

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
            "contexts": retrieved_documents,
            "execution_time": execution_time,
            "metadata": {
                "num_retrieved": len(retrieved_documents),
                "processing_time": execution_time,
                "processing_time_ms": execution_time_ms,
                "execution_time": execution_time,
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

    def _ensure_similarity_scores(self, documents: List[Any]) -> List[Document]:
        """Ensure all documents carry a similarity_score for contract tests."""
        normalized: List[Document] = []
        for doc in documents:
            if isinstance(doc, Document):
                score = (
                    doc.metadata.get("similarity_score")
                    if hasattr(doc, "metadata")
                    else None
                )
                score_value = float(score) if score is not None else 1.0
                normalized.append(
                    Document(
                        id=doc.id,
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "similarity_score": score_value},
                    )
                )
            else:
                normalized.append(
                    Document(
                        id="fallback_context",
                        page_content=str(doc),
                        metadata={"similarity_score": 1.0},
                    )
                )
        return normalized

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
                logger.warning(
                    "Text search returned 0 results. Falling back to vector search."
                )
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
                logger.warning(
                    "HNSW vector search returned 0 results. Falling back to IRISVectorStore."
                )
                fallback_docs = self._fallback_to_vector_search(query_text, top_k)
                return fallback_docs, "vector_fallback"
            return documents, method
        except Exception as e:
            logger.error(f"HNSW vector retrieval failed: {e}")
            logger.info("Falling back to IRISVectorStore vector search")
            fallback_docs = self._fallback_to_vector_search(query_text, top_k)
            return fallback_docs, "vector_fallback"

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

    def _extract_sources(self, documents: List[Document]) -> List[str]:
        return super()._extract_sources(documents)

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
        graph_core_path = None
        discoverer = getattr(self.discovery, "discover_graph_core_path", None)
        if callable(discoverer):
            discovered_path = discoverer()
            graph_core_path = str(discovered_path) if discovered_path else None
        return {
            "hybrid_enabled": self.is_hybrid_enabled(),
            "iris_engine_available": self.iris_engine is not None,
            "fusion_engine_available": self.fusion_engine is not None,
            "text_engine_available": self.text_engine is not None,
            "vector_optimizer_available": self.vector_optimizer is not None,
            "graph_core_path": graph_core_path,
        }
