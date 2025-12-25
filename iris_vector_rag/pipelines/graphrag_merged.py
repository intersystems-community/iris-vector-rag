import time
"""
Ontology-Enhanced GraphRAG Pipeline - Merged production-hardened + complete functionality.

This implementation combines:
- Production-hardened error handling and validation from current version
- Complete entity extraction and ingestion pipeline from old version
- Advanced ontology support for domain-specific knowledge representation
- Ontology-aware query expansion and entity reasoning
- Configurable vector fallback for robustness
- Performance monitoring and graph traversal optimization
- Integration with PRefLexOR-style hybrid ontology approach
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..config.manager import ConfigurationManager
from ..core.base import RAGPipeline
from ..core.connection import ConnectionManager
from ..core.exceptions import RAGException
from ..core.models import Document
from ..embeddings.manager import EmbeddingManager

# Import general-purpose ontology components
from ..ontology.plugins import (
    GeneralOntologyPlugin,
    create_plugin_from_config,
    get_ontology_plugin,
)
from ..ontology.reasoner import OntologyReasoner, QueryExpander
from ..services.entity_extraction import EntityExtractionService
from ..visualization.graph_visualizer import (
    GraphVisualizationException,
    GraphVisualizer,
)

logger = logging.getLogger(__name__)


class GraphRAGException(RAGException):
    """Exception raised when GraphRAG operations fail."""

    pass


class KnowledgeGraphNotPopulatedException(GraphRAGException):
    """Exception raised when knowledge graph is not populated with entities."""

    pass


class EntityExtractionFailedException(GraphRAGException):
    """Exception raised when entity extraction fails during document loading."""

    pass


class GraphRAGPipeline(RAGPipeline):
    """
    Merged GraphRAG pipeline combining production-hardened validation with complete functionality.

    Features:
    - Fail-hard validation with clear error messages
    - Complete entity extraction and relationship mapping
    - Graph-based retrieval with query entity extraction
    - Optional vector fallback for robustness
    - Performance monitoring and debug instrumentation
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        vector_store=None,
    ):
        if connection_manager is None:
            connection_manager = ConnectionManager()
        if config_manager is None:
            config_manager = ConfigurationManager()

        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func
        self.embedding_manager = EmbeddingManager(config_manager)

        # Initialize entity extraction service (primary) with local fallback
        try:
            self.entity_extraction_service = EntityExtractionService(
                config_manager=config_manager,
                connection_manager=connection_manager,
                embedding_manager=self.embedding_manager,
            )
            self.use_service_extraction = True
        except Exception as e:
            logger.warning(
                f"EntityExtractionService unavailable, using local extraction: {e}"
            )
            self.entity_extraction_service = None
            self.use_service_extraction = False

        # Configuration
        self.pipeline_config = self.config_manager.get("pipelines:graphrag", {})
        self.ontology_config = self.config_manager.get("ontology", {})
        self.default_top_k = self.pipeline_config.get("default_top_k", 10)
        self.max_depth = self.pipeline_config.get("max_depth", 2)
        self.max_entities = self.pipeline_config.get("max_entities", 50)
        self.enable_vector_fallback = self.pipeline_config.get(
            "enable_vector_fallback", True
        )

        # Initialize general-purpose ontology support
        self.ontology_enabled = self.ontology_config.get("enabled", False)
        self.ontology_plugin = None
        self.reasoner = None
        self.query_expander = None

        if self.ontology_enabled:
            self._init_ontology_support()

        # Initialize visualization support
        try:
            self.graph_visualizer = GraphVisualizer(
                connection_manager=connection_manager, config_manager=config_manager
            )
            self.visualization_enabled = True
        except Exception as e:
            logger.warning(f"Graph visualization unavailable: {e}")
            self.graph_visualizer = None
            self.visualization_enabled = False

        # Store traversal data for visualization
        self.last_traversal_data = {}

        logger.info(
            f"GraphRAG pipeline initialized - service_extraction={self.use_service_extraction}, fallback={self.enable_vector_fallback}, visualization={self.visualization_enabled}, ontology={self.ontology_enabled}"
        )

    def _init_ontology_support(self) -> None:
        """Initialize general-purpose ontology plugin and reasoning capabilities."""
        try:
            # Create general-purpose ontology plugin from configuration
            if self.ontology_config.get("sources"):
                # Load from configured sources
                self.ontology_plugin = create_plugin_from_config(self.ontology_config)
            else:
                # Create empty plugin that can be loaded dynamically
                self.ontology_plugin = GeneralOntologyPlugin()
                self.ontology_plugin.auto_detect_domain = self.ontology_config.get(
                    "auto_detect_domain", True
                )

            # Initialize reasoner if ontology has concepts
            if self.ontology_plugin and self.ontology_plugin.concepts:
                self.reasoner = OntologyReasoner(self.ontology_plugin.hierarchy)

                # Initialize query expander
                self.query_expander = QueryExpander(self.ontology_plugin.hierarchy)

                detected_domain = self.ontology_plugin.domain
                concept_count = len(self.ontology_plugin.concepts)
                logger.info(
                    f"Ontology support initialized: domain={detected_domain}, concepts={concept_count}"
                )
            else:
                logger.info("Ontology plugin created but no concepts loaded")

        except Exception as e:
            logger.error(f"Failed to initialize ontology support: {e}")
            self.ontology_enabled = False
            self.ontology_plugin = None

    def query_with_reasoning(
        self, query: str, use_reasoning: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced query method with ontology-based reasoning and expansion.

        Args:
            query: User query
            use_reasoning: Whether to apply ontology reasoning
            **kwargs: Additional query parameters

        Returns:
            Enhanced query results with ontology insights
        """
        start_time = time.time()

        try:
            # Apply query expansion if ontology support is enabled
            expanded_query = query
            expansion_info = {}

            if self.ontology_enabled and use_reasoning and self.query_expander:
                expansion_result = self.query_expander.expand_query(
                    query,
                    strategy=self.ontology_config.get("expansion_strategy", "synonyms"),
                )
                expanded_query = expansion_result.expanded_query
                expansion_info = {
                    "original_query": expansion_result.original_query,
                    "expanded_query": expansion_result.expanded_query,
                    "expansion_terms": expansion_result.expansion_terms,
                    "semantic_concepts": expansion_result.semantic_concepts,
                    "confidence": expansion_result.confidence,
                }

                logger.debug(f"Query expanded: {query} -> {expanded_query}")

            # Execute enhanced query
            base_results = self.query(expanded_query, **kwargs)

            # Add ontology reasoning if enabled
            if self.ontology_enabled and use_reasoning:
                ontology_insights = self._get_ontology_insights(query, base_results)
                base_results["ontology_insights"] = ontology_insights
                base_results["query_expansion"] = expansion_info

            # Add performance metrics
            base_results["processing_time"] = time.time() - start_time
            base_results["ontology_enabled"] = self.ontology_enabled

            return base_results

        except Exception as e:
            logger.error(f"Enhanced query failed: {e}")
            # Fallback to basic query
            return self.query(query, **kwargs)

    def _get_ontology_insights(
        self, query: str, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate ontology-based insights for query results using general-purpose ontology."""
        insights = {
            "domain": "general",
            "detected_domain": None,
            "inferred_concepts": [],
            "semantic_relationships": [],
            "confidence_scores": {},
        }

        if not self.ontology_plugin or not self.reasoner:
            return insights

        try:
            query_lower = query.lower()

            # Get detected domain from ontology plugin
            if hasattr(self.ontology_plugin, "domain"):
                insights["detected_domain"] = self.ontology_plugin.domain

            # Calculate domain relevance scoring
            domain_score = 0
            concept_matches = []

            for concept in self.reasoner.hierarchy.concepts.values():
                if concept.label.lower() in query_lower:
                    domain_score += 1
                    concept_matches.append(concept)

                for synonym in concept.get_all_synonyms():
                    if synonym in query_lower:
                        domain_score += 0.5
                        if concept not in concept_matches:
                            concept_matches.append(concept)

            # Generate inferred concepts from matched concepts
            for concept in concept_matches:
                # Get related concepts
                ancestors = self.reasoner.hierarchy.get_ancestors(
                    concept.id, max_depth=2
                )
                descendants = self.reasoner.hierarchy.get_descendants(
                    concept.id, max_depth=1
                )

                insights["inferred_concepts"].append(
                    {
                        "concept_id": concept.id,
                        "label": concept.label,
                        "description": concept.description,
                        "ancestors": len(ancestors),
                        "descendants": len(descendants),
                        "domain": insights["detected_domain"],
                    }
                )

            # Add semantic relationships between concepts
            if len(concept_matches) > 1:
                for i, concept1 in enumerate(concept_matches):
                    for concept2 in concept_matches[i + 1 :]:
                        # Check if concepts are related
                        if concept1.id in self.reasoner.hierarchy.get_ancestors(
                            concept2.id
                        ):
                            insights["semantic_relationships"].append(
                                {
                                    "source": concept1.label,
                                    "target": concept2.label,
                                    "relationship": "ancestor_of",
                                }
                            )
                        elif concept2.id in self.reasoner.hierarchy.get_ancestors(
                            concept1.id
                        ):
                            insights["semantic_relationships"].append(
                                {
                                    "source": concept2.label,
                                    "target": concept1.label,
                                    "relationship": "ancestor_of",
                                }
                            )

            # Add confidence scores
            insights["confidence_scores"] = {
                "domain_relevance": min(1.0, domain_score * 0.1),
                "concept_inference": min(1.0, len(insights["inferred_concepts"]) * 0.2),
                "relationship_inference": min(
                    1.0, len(insights["semantic_relationships"]) * 0.3
                ),
                "overall": min(
                    1.0,
                    (
                        domain_score
                        + len(insights["inferred_concepts"])
                        + len(insights["semantic_relationships"])
                    )
                    * 0.05,
                ),
            }

        except Exception as e:
            logger.error(f"Error generating ontology insights: {e}")
            insights["error"] = str(e)

        return insights

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Load documents with complete entity extraction and graph building."""
        start_time = time.time()

        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            documents = self._load_documents_from_path(documents_path)

        if not documents:
            raise GraphRAGException("No documents found to load")

        # Store documents first for vector search compatibility
        generate_embeddings = kwargs.get("generate_embeddings", True)
        if generate_embeddings and self.vector_store:
            self.vector_store.add_documents(documents, auto_chunk=True)
        else:
            self._store_documents(documents)

        # Extract entities using service or local extraction
        if self.use_service_extraction:
            total_entities, total_relationships = self._extract_via_service(documents)
        else:
            total_entities, total_relationships = self._extract_locally(documents)

        # Validate knowledge graph population
        if total_entities == 0:
            raise KnowledgeGraphNotPopulatedException(
                "No entities were extracted from documents. Knowledge graph is empty."
            )

        processing_time = time.time() - start_time
        logger.info(
            f"GraphRAG: Loaded {len(documents)} documents with {total_entities} entities "
            f"and {total_relationships} relationships in {processing_time:.2f}s"
        )

    def _extract_via_service(self, documents: List[Document]) -> Tuple[int, int]:
        """Extract entities using EntityExtractionService."""
        total_entities = 0
        total_relationships = 0
        failed_documents = []

        for doc in documents:
            try:
                result = self.entity_extraction_service.process_document(doc)
                if not result.get("stored", False):
                    raise EntityExtractionFailedException(
                        f"Failed to store entities for document {doc.id}"
                    )

                total_entities += result["entities_count"]
                total_relationships += result["relationships_count"]

            except Exception as e:
                failed_documents.append(doc.id)
                logger.error(
                    f"Service entity extraction failed for document {doc.id}: {e}"
                )

        if failed_documents:
            raise EntityExtractionFailedException(
                f"Entity extraction failed for {len(failed_documents)} documents: {failed_documents}"
            )

        return total_entities, total_relationships

    def _extract_locally(self, documents: List[Document]) -> Tuple[int, int]:
        """Extract entities using local extraction methods."""
        total_entities = 0
        total_relationships = 0

        for doc in documents:
            try:
                entities = self._extract_entities(doc)
                relationships = self._extract_relationships(doc, entities)

                self._store_entities(doc.id, entities)
                self._store_relationships(doc.id, relationships)

                total_entities += len(entities)
                total_relationships += len(relationships)

            except Exception as e:
                logger.error(
                    f"Local entity extraction failed for document {doc.id}: {e}"
                )
                raise EntityExtractionFailedException(
                    f"Local extraction failed for document {doc.id}: {e}"
                )

        return total_entities, total_relationships

    def query(self, query_text: str, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        """Execute GraphRAG query with knowledge graph traversal and optional visualization."""
        start_time = time.time()
        start_perf = time.perf_counter()

        include_sources = kwargs.get("include_sources", True)
        custom_prompt = kwargs.get("custom_prompt")
        generate_answer = kwargs.get("generate_answer", True)

        # Visualization parameters
        visualize = kwargs.get("visualize", False)
        visualization_type = kwargs.get(
            "visualization_type", "plotly"
        )  # plotly, d3, or traversal
        highlight_path = kwargs.get("highlight_path", [])

        # Validate knowledge graph is populated
        self._validate_knowledge_graph()

        # Try graph-based retrieval
        try:
            retrieved_documents, method = self._retrieve_via_kg(query_text, top_k)
        except GraphRAGException as e:
            if self.enable_vector_fallback:
                logger.warning(f"Graph retrieval failed, trying vector fallback: {e}")
                retrieved_documents, method = self._vector_fallback_retrieval(
                    query_text, top_k
                )
            else:
                raise

        # Generate answer
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
                "pipeline_type": "graphrag_merged",
                "retrieval_method": method,
                "generated_answer": generate_answer and answer is not None,
            },
        }

        # Attach performance instrumentation
        if hasattr(self, "_debug_db_execs"):
            response["metadata"]["db_exec_count"] = int(self._debug_db_execs)
        if hasattr(self, "_debug_step_times"):
            response["metadata"]["step_timings_ms"] = dict(self._debug_step_times)

        if include_sources:
            response["sources"] = self._extract_sources(retrieved_documents)

        # Generate visualization if requested
        if visualize and self.visualization_enabled and self.last_traversal_data:
            try:
                visualization_html = self._generate_visualization(
                    query_text, visualization_type, highlight_path, response
                )
                response["visualization"] = visualization_html
                response["metadata"]["visualization_generated"] = True
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")
                response["metadata"]["visualization_error"] = str(e)

        logger.info(
            f"GraphRAG query completed in {execution_time:.2f}s ({execution_time_ms:.1f}ms) - "
            f"{len(retrieved_documents)} docs via {method}"
        )
        return response

    def _validate_knowledge_graph(self) -> None:
        """Validate that the knowledge graph has entities before allowing queries."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            entity_count = cursor.fetchone()[0]

            if entity_count == 0:
                raise KnowledgeGraphNotPopulatedException(
                    "Knowledge graph is empty. Load documents with entity extraction before querying."
                )

            logger.debug(
                f"Knowledge graph validation passed: {entity_count} entities found"
            )

        except Exception as e:
            if isinstance(e, KnowledgeGraphNotPopulatedException):
                raise
            raise GraphRAGException(f"Knowledge graph validation failed: {e}")
        finally:
            cursor.close()

    def _retrieve_via_kg(
        self, query_text: str, top_k: int
    ) -> Tuple[List[Document], str]:
        """Retrieve documents via knowledge graph traversal with query entity extraction."""
        # Initialize performance monitoring
        self._debug_db_execs = 0
        self._debug_step_times = {}

        # Extract entities from query
        t0 = time.perf_counter()
        query_entities = self._extract_query_entities(query_text)
        self._debug_step_times["query_entity_extraction_ms"] = (
            time.perf_counter() - t0
        ) * 1000.0

        if not query_entities:
            # Fallback to keyword-based seed entity finding
            t1 = time.perf_counter()
            seed_entities = self._find_seed_entities(query_text)
            self._debug_step_times["find_seed_entities_ms"] = (
                time.perf_counter() - t1
            ) * 1000.0
        else:
            # Convert query entities to seed entities format
            seed_entities = [
                (f"query_entity_{i}", entity, 0.9)
                for i, entity in enumerate(query_entities)
            ]
            self._debug_step_times["find_seed_entities_ms"] = 0.0

        # Traverse graph
        t2 = time.perf_counter()
        relevant_entities = self._traverse_graph(seed_entities)
        self._debug_step_times["traverse_graph_ms"] = (
            time.perf_counter() - t2
        ) * 1000.0

        # Get documents
        t3 = time.perf_counter()
        docs = self._get_documents_from_entities(relevant_entities, top_k)
        self._debug_step_times["get_documents_ms"] = (time.perf_counter() - t3) * 1000.0

        return docs, "knowledge_graph_traversal"

    def _extract_query_entities(self, query_text: str) -> List[str]:
        """Extract entities from query text by matching against known entities."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Get known entities
            cursor.execute("SELECT DISTINCT entity_name FROM RAG.Entities")
            known_entities = [row[0].lower() for row in cursor.fetchall()]

            # Find matches in query
            query_lower = query_text.lower()
            found_entities = []

            for entity in known_entities:
                if entity in query_lower:
                    found_entities.append(entity)

            # Partial word matching if no exact matches
            if not found_entities:
                words = query_lower.split()
                for word in words:
                    if len(word) > 3:
                        for entity in known_entities:
                            if word in entity or entity in word:
                                found_entities.append(entity)
                                break

            self._debug_db_execs = getattr(self, "_debug_db_execs", 0) + 1
            return list(set(found_entities))[:10]  # Limit and deduplicate

        except Exception as e:
            logger.warning(f"Query entity extraction failed: {e}")
            return []
        finally:
            cursor.close()

    def _find_seed_entities(self, query_text: str) -> List[Tuple[str, str, float]]:
        """Find seed entities using keyword matching in RAG.Entities table."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            query_keywords = query_text.lower().split()[:5]
            if not query_keywords:
                raise GraphRAGException("Query contains no searchable keywords")

            conditions = []
            params = []
            for keyword in query_keywords:
                conditions.append("LOWER(entity_name) LIKE ?")
                params.append(f"%{keyword}%")

            query = f"""
                SELECT TOP 10 entity_id, entity_name, entity_type
                FROM RAG.Entities
                WHERE {' OR '.join(conditions)}
                  AND entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
            """

            self._debug_db_execs = getattr(self, "_debug_db_execs", 0) + 1
            cursor.execute(query, params)
            results = cursor.fetchall()

            seed_entities = []
            for row in results:
                try:
                    if isinstance(row, (list, tuple)):
                        if len(row) >= 2:
                            entity_id, entity_name = row[0], row[1]
                        elif len(row) == 1:
                            entity_id, entity_name = row[0], str(row[0])
                        else:
                            continue
                    else:
                        entity_id, entity_name = row, str(row)
                    seed_entities.append((str(entity_id), str(entity_name), 0.9))
                except Exception:
                    continue

            if not seed_entities:
                raise GraphRAGException(
                    f"No seed entities found for query '{query_text}'"
                )

            return seed_entities

        except Exception as e:
            if isinstance(e, GraphRAGException):
                raise
            raise GraphRAGException(f"Database error finding seed entities: {e}")
        finally:
            cursor.close()

    def _traverse_graph(self, seed_entities: List[Tuple[str, str, float]]) -> Set[str]:
        """Traverse knowledge graph using RAG.EntityRelationships and store data for visualization."""
        if not seed_entities:
            raise GraphRAGException("No seed entities provided for graph traversal")

        relevant_entities: Set[str] = {e[0] for e in seed_entities}
        current_entities: Set[str] = {e[0] for e in seed_entities}

        # Store traversal data for visualization
        self.last_traversal_data = {
            "seed_entities": seed_entities,
            "traversal_result": set(),
            "relationships": [],
        }

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            for depth in range(self.max_depth):
                if len(relevant_entities) >= self.max_entities or not current_entities:
                    break

                entity_list = list(current_entities)
                placeholders = ",".join(["?" for _ in entity_list])

                query = f"""
                    SELECT DISTINCT r.target_entity_id
                    FROM RAG.EntityRelationships r
                    WHERE r.source_entity_id IN ({placeholders})
                    UNION
                    SELECT DISTINCT r.source_entity_id
                    FROM RAG.EntityRelationships r
                    WHERE r.target_entity_id IN ({placeholders})
                """

                self._debug_db_execs = getattr(self, "_debug_db_execs", 0) + 1
                cursor.execute(query, entity_list + entity_list)
                results = cursor.fetchall()

                next_entities = set()
                for (entity_id,) in results:
                    entity_id_str = str(entity_id)
                    if entity_id_str not in relevant_entities:
                        relevant_entities.add(entity_id_str)
                        next_entities.add(entity_id_str)

                current_entities = next_entities

            # Store final traversal results for visualization
            self.last_traversal_data["traversal_result"] = relevant_entities

            # Get relationships for visualization
            if self.visualization_enabled:
                self.last_traversal_data["relationships"] = (
                    self._get_traversal_relationships(relevant_entities)
                )

        except Exception as e:
            raise GraphRAGException(f"Database error traversing graph: {e}")
        finally:
            cursor.close()

        if len(relevant_entities) == len(seed_entities):
            raise GraphRAGException("Graph traversal found no additional entities")

        return relevant_entities

    def _get_documents_from_entities(
        self, entity_ids: Set[str], top_k: int
    ) -> List[Document]:
        """Get documents associated with entities."""
        if not entity_ids:
            raise GraphRAGException("No entity IDs provided for document retrieval")

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            entity_list = list(entity_ids)[:50]
            placeholders = ",".join(["?" for _ in entity_list])

            query = f"""
                SELECT DISTINCT sd.doc_id, sd.text_content, sd.title
                FROM RAG.SourceDocuments sd
                JOIN RAG.Entities e ON sd.doc_id = e.source_doc_id
                WHERE e.entity_id IN ({placeholders})
                ORDER BY sd.doc_id
            """

            self._debug_db_execs = getattr(self, "_debug_db_execs", 0) + 1
            cursor.execute(query, entity_list)
            results = cursor.fetchall()

            if not results:
                raise GraphRAGException(
                    f"No documents found for {len(entity_list)} entities"
                )

            docs = []
            seen_ids = set()
            for doc_id, content, title in results:
                doc_id_str = str(doc_id)
                if doc_id_str not in seen_ids:
                    seen_ids.add(doc_id_str)
                    content_str = self._read_iris_data(content)
                    title_str = self._read_iris_data(title)

                    docs.append(
                        Document(
                            id=doc_id_str,
                            page_content=content_str,
                            metadata={
                                "title": title_str,
                                "retrieval_method": "knowledge_graph",
                            },
                        )
                    )

                    if len(docs) >= top_k:
                        break

            return docs

        except Exception as e:
            if isinstance(e, GraphRAGException):
                raise
            raise GraphRAGException(f"Database error getting documents: {e}")
        finally:
            cursor.close()

    def _vector_fallback_retrieval(
        self, query_text: str, top_k: int
    ) -> Tuple[List[Document], str]:
        """Fallback to vector search if graph retrieval fails."""
        if not self.vector_store:
            raise GraphRAGException(
                "Vector fallback requested but no vector store available"
            )

        logger.info("Performing vector fallback retrieval")
        try:
            docs = self.vector_store.similarity_search(query_text, k=top_k)
            for doc in docs:
                if doc.metadata:
                    doc.metadata["retrieval_method"] = "vector_fallback"
            return docs, "vector_fallback"
        except Exception as e:
            raise GraphRAGException(f"Vector fallback retrieval failed: {e}")

    def _extract_entities(self, document: Document) -> List[Dict[str, Any]]:
        """Extract entities from document text using simple keyword extraction."""
        text = document.page_content
        words = text.split()
        entities = []

        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 3:
                entity_embedding = self.embedding_manager.embed_text(word)
                entities.append(
                    {
                        "entity_id": f"{document.id}_entity_{i}",
                        "entity_text": word,
                        "entity_type": "KEYWORD",
                        "position": i,
                        "embedding": entity_embedding,
                    }
                )

        return entities[: self.max_entities]

    def _extract_relationships(
        self, document: Document, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities based on co-occurrence."""
        relationships = []

        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                pos_diff = abs(entity1["position"] - entity2["position"])
                if pos_diff <= 10:
                    relationships.append(
                        {
                            "relationship_id": f"{document.id}_rel_{i}_{j}",
                            "source_entity": entity1["entity_id"],
                            "target_entity": entity2["entity_id"],
                            "relationship_type": "CO_OCCURS",
                            "strength": 1.0 / (pos_diff + 1),
                        }
                    )

        return relationships

    def _store_entities(self, document_id: str, entities: List[Dict[str, Any]]):
        """Store entities in the database."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            for entity in entities:
                cursor.execute(
                    """
                    INSERT INTO RAG.Entities
                    (entity_id, entity_name, entity_type, source_doc_id)
                    VALUES (?, ?, ?, ?)
                """,
                    [
                        entity["entity_id"],
                        entity["entity_text"],
                        entity["entity_type"],
                        document_id,
                    ],
                )

            connection.commit()
            logger.debug(f"Stored {len(entities)} entities for document {document_id}")

        except Exception as e:
            connection.rollback()
            logger.error(f"Failed to store entities for document {document_id}: {e}")
            raise
        finally:
            cursor.close()

    def _store_relationships(
        self, document_id: str, relationships: List[Dict[str, Any]]
    ):
        """Store relationships in the database."""
        if not relationships:
            return

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            for rel in relationships:
                cursor.execute(
                    """
                    INSERT INTO RAG.EntityRelationships
                    (relationship_id, source_entity_id, target_entity_id, relationship_type, confidence_score, source_doc_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        rel["relationship_id"],
                        rel["source_entity"],
                        rel["target_entity"],
                        rel["relationship_type"],
                        rel["strength"],
                        document_id,
                    ],
                )

            connection.commit()
            logger.debug(
                f"Stored {len(relationships)} relationships for document {document_id}"
            )

        except Exception as e:
            connection.rollback()
            logger.error(
                f"Failed to store relationships for document {document_id}: {e}"
            )
            raise
        finally:
            cursor.close()

    def _load_documents_from_path(self, documents_path: str) -> List[Document]:
        """Load documents from file or directory path."""
        import os

        documents = []
        if os.path.isfile(documents_path):
            documents.append(self._load_single_file(documents_path))
        elif os.path.isdir(documents_path):
            for filename in os.listdir(documents_path):
                file_path = os.path.join(documents_path, filename)
                if os.path.isfile(file_path):
                    try:
                        documents.append(self._load_single_file(file_path))
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
        return documents

    def _load_single_file(self, file_path: str) -> Document:
        """Load a single file as a Document."""
        import os

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
        }
        return Document(page_content=content, metadata=metadata)

    def _read_iris_data(self, data) -> str:
        """Handle IRIS stream data."""
        if data is None:
            return ""
        try:
            import jaydebeapi

            connection = self.connection_manager.get_connection()
            if hasattr(connection, "__class__") and "jaydebeapi" in str(
                connection.__class__
            ):
                if hasattr(data, "read"):
                    return data.read().decode("utf-8") if data else ""
        except ImportError:
            pass
        return str(data or "")

    def _generate_answer(
        self, query: str, documents: List[Document], custom_prompt: Optional[str] = None
    ) -> str:
        """Generate answer using LLM."""
        if not documents:
            return "No relevant documents found to answer the query."

        context_parts = []
        for doc in documents[:5]:
            doc_content = str(doc.page_content or "")[:1000]
            title = (
                doc.metadata.get("title", "Untitled") if doc.metadata else "Untitled"
            )
            context_parts.append(f"Document {doc.id} ({title}):\n{doc_content}")

        context = "\n\n".join(context_parts)

        if custom_prompt:
            prompt = custom_prompt.format(query=query, context=context)
        else:
            prompt = f"""Based on the knowledge graph context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        try:
            return self.llm_func(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information."""
        sources = []
        for doc in documents:
            sources.append(
                {
                    "document_id": doc.id,
                    "source": (
                        doc.metadata.get("source", "Unknown")
                        if doc.metadata
                        else "Unknown"
                    ),
                    "title": (
                        doc.metadata.get("title", "Unknown")
                        if doc.metadata
                        else "Unknown"
                    ),
                    "retrieval_method": (
                        doc.metadata.get("retrieval_method", "unknown")
                        if doc.metadata
                        else "unknown"
                    ),
                }
            )
        return sources

    def retrieve(self, query_text: str, top_k: int = 10, **kwargs) -> List[Document]:
        """Get documents only."""
        result = self.query(query_text, top_k=top_k, generate_answer=False, **kwargs)
        return result["retrieved_documents"]

    def _get_traversal_relationships(
        self, entity_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """Get relationships between entities for visualization."""
        if not entity_ids:
            return []

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            entity_list = list(entity_ids)
            placeholders = ",".join(["?" for _ in entity_list])

            query = f"""
                SELECT source_entity_id, target_entity_id, relationship_type,
                       confidence_score, source_doc_id
                FROM RAG.EntityRelationships
                WHERE source_entity_id IN ({placeholders})
                   AND target_entity_id IN ({placeholders})
            """

            cursor.execute(query, entity_list + entity_list)
            results = cursor.fetchall()

            relationships = []
            for source_id, target_id, rel_type, confidence, doc_id in results:
                relationships.append(
                    {
                        "source_entity_id": str(source_id),
                        "target_entity_id": str(target_id),
                        "relationship_type": str(rel_type),
                        "confidence_score": float(confidence) if confidence else 0.0,
                        "source_doc_id": str(doc_id),
                    }
                )

            return relationships

        except Exception as e:
            logger.error(f"Failed to get traversal relationships: {e}")
            return []
        finally:
            cursor.close()

    def _generate_visualization(
        self,
        query_text: str,
        visualization_type: str,
        highlight_path: List[str],
        query_result: Dict[str, Any],
    ) -> str:
        """Generate visualization HTML for the query results."""
        try:
            if visualization_type == "traversal":
                # Generate traversal path visualization
                return self.graph_visualizer.visualize_traversal_path(query_result)

            elif visualization_type in ["plotly", "d3"]:
                # Build graph from traversal data
                if not self.last_traversal_data:
                    raise GraphVisualizationException(
                        "No traversal data available for visualization"
                    )

                graph = self.graph_visualizer.build_graph_from_traversal(
                    self.last_traversal_data["seed_entities"],
                    self.last_traversal_data["traversal_result"],
                )

                if visualization_type == "plotly":
                    return self.graph_visualizer.generate_plotly_visualization(
                        graph, highlight_path
                    )
                else:  # d3
                    return self.graph_visualizer.generate_d3_visualization(graph)

            else:
                raise GraphVisualizationException(
                    f"Unknown visualization type: {visualization_type}"
                )

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise GraphVisualizationException(
                f"Failed to generate {visualization_type} visualization: {e}"
            )

    def ask(self, question: str, **kwargs) -> str:
        """Get answer only."""
        result = self.query(question, **kwargs)
        return result.get("answer", "No answer generated")
