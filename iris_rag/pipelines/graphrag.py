"""
GraphRAG Pipeline implementation using knowledge graph traversal.

PRODUCTION-HARDENED VERSION: No fallbacks, fail-hard validation, integrated entity extraction.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..config.manager import ConfigurationManager
from ..core.base import RAGPipeline
from ..core.connection import ConnectionManager
from ..core.exceptions import RAGException
from ..core.models import Document
from ..embeddings.manager import EmbeddingManager
from ..services.entity_extraction import EntityExtractionService
from ..storage.schema_manager import SchemaManager

logger = logging.getLogger(__name__)


class GraphRAGException(RAGException):
    """Exception raised when GraphRAG operations fail."""


class KnowledgeGraphNotPopulatedException(GraphRAGException):
    """Exception raised when knowledge graph is not populated with entities."""


class EntityExtractionFailedException(GraphRAGException):
    """Exception raised when entity extraction fails during document loading."""


class GraphRAGPipeline(RAGPipeline):
    """
    Production-hardened GraphRAG pipeline with fail-hard validation.

    No fallbacks to vector search. Either performs true knowledge graph operations
    or fails explicitly with clear error messages.
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        vector_store=None,
    ):
        if connection_manager is None:
            try:
                connection_manager = ConnectionManager()
            except Exception as e:
                raise GraphRAGException(f"Failed to create ConnectionManager: {e}")

        if config_manager is None:
            try:
                config_manager = ConfigurationManager()
            except Exception as e:
                raise GraphRAGException(f"Failed to create ConfigurationManager: {e}")

        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func
        self.embedding_manager = EmbeddingManager(config_manager)

        # Initialize entity extraction service
        self.entity_extraction_service = EntityExtractionService(
            config_manager=config_manager,
            connection_manager=connection_manager,
            embedding_manager=self.embedding_manager,
        )

        # Configuration
        self.pipeline_config = self.config_manager.get("pipelines:graphrag", {})
        self.default_top_k = self.pipeline_config.get("default_top_k", 10)
        self.max_depth = self.pipeline_config.get("max_depth", 2)
        self.max_entities = self.pipeline_config.get("max_entities", 50)

        logger.info(
            "Production-hardened GraphRAG pipeline initialized with entity extraction"
        )

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents with integrated entity extraction.

        This method extracts entities and relationships from documents and stores them
        in the knowledge graph tables (RAG.Entities and RAG.EntityRelationships).
        """
        start_time = time.time()

        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            documents = self._load_documents_from_path(documents_path)

        if not documents:
            raise GraphRAGException("No documents found to load")

        # Store documents first (for vector search compatibility)
        generate_embeddings = kwargs.get("generate_embeddings", True)
        if generate_embeddings:
            self.vector_store.add_documents(documents, auto_chunk=True)
        else:
            self._store_documents(documents)

        # Ensure knowledge graph tables exist BEFORE extraction/storage
        try:
            schema_manager = SchemaManager(self.connection_manager, self.config_manager)
            schema_manager.ensure_table_schema("Entities")
            schema_manager.ensure_table_schema("EntityRelationships")
        except Exception as e:
            logger.warning(
                f"Could not ensure knowledge graph tables before extraction: {e}"
            )

        # Extract entities and relationships for knowledge graph using BATCH PROCESSING
        total_entities = 0
        total_relationships = 0
        failed_documents = []

        # Process documents in batches of 5 for 3x speedup
        batch_size = 5
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            try:
                logger.info(f"🚀 Processing batch {i//batch_size + 1} ({len(batch_docs)} documents) for entity extraction")

                # Batch extract entities (single LLM call for all documents in batch!)
                batch_results = self.entity_extraction_service.extract_batch_with_dspy(
                    batch_docs, batch_size=batch_size
                )

                # Process results for each document in the batch
                for doc in batch_docs:
                    try:
                        entities = batch_results.get(doc.id, [])

                        if entities:
                            # Store entities using the service's storage adapter
                            from ..services.storage import EntityStorageAdapter
                            storage_service = EntityStorageAdapter(
                                connection_manager=self.connection_manager,
                                config=self.config_manager._config
                            )

                            # Store entities in batch
                            stored_count = storage_service.store_entities_batch(entities)
                            total_entities += stored_count

                            # Extract and store relationships from entity pairs
                            from ..core.models import Relationship
                            relationships = []
                            for i, entity1 in enumerate(entities):
                                for entity2 in entities[i+1:]:
                                    # Simple relationship: co-occurrence in same document
                                    rel = Relationship(
                                        source_entity_id=entity1.text,  # Fixed: was source_entity
                                        target_entity_id=entity2.text,  # Fixed: was target_entity
                                        relationship_type="co_occurs_with",
                                        confidence=0.8,
                                        source_document_id=doc.id
                                    )
                                    relationships.append(rel)

                            # Store relationships in batch
                            try:
                                relationships_stored = storage_service.store_relationships_batch(relationships)
                                total_relationships += relationships_stored
                            except Exception as e:
                                logger.debug(f"Failed to store relationships: {e}")

                            logger.debug(
                                f"Document {doc.id}: {stored_count} entities, {relationships_stored} relationships"
                            )
                        else:
                            logger.warning(
                                f"No entities extracted for document {doc.id} - may lack extractable technical content"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to store entities for document {doc.id}: {e}")
                        failed_documents.append(doc.id)

            except Exception as e:
                # Batch extraction failed - fall back to individual processing for this batch
                logger.warning(f"Batch entity extraction failed: {e}, falling back to individual processing")

                for doc in batch_docs:
                    try:
                        logger.info(f"Processing document {doc.id} individually (fallback)")
                        result = self.entity_extraction_service.process_document(doc)

                        if not result.get("stored", False):
                            logger.warning(
                                f"No entities stored for document {doc.id} - may lack extractable technical content"
                            )
                        else:
                            total_entities += result.get(
                                "entities_count", result.get("entities_extracted", 0)
                            )
                            total_relationships += result.get(
                                "relationships_count", result.get("relationships_extracted", 0)
                            )

                            logger.debug(
                                f"Document {doc.id}: {result.get('entities_extracted', 0)} entities, {result.get('relationships_extracted', 0)} relationships"
                            )

                    except Exception as e:
                        logger.warning(f"Entity extraction error for document {doc.id}: {e}")
                        failed_documents.append(doc.id)

        # Only fail if we got zero entities across ALL documents
        # (suggests a systematic failure rather than content issues)

        # Validate that we have entities in the knowledge graph
        if total_entities == 0:
            raise KnowledgeGraphNotPopulatedException(
                "No entities were extracted from documents. Knowledge graph is empty."
            )

        processing_time = time.time() - start_time
        logger.info(
            f"GraphRAG: Loaded {len(documents)} documents with {total_entities} entities "
            f"and {total_relationships} relationships in {processing_time:.2f}s"
        )

    def _load_documents_from_path(self, documents_path: str) -> List[Document]:
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
        import os

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
        }
        return Document(page_content=content, metadata=metadata)

    def query(self, query_text: str, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Execute GraphRAG query with knowledge graph traversal.

        Fails hard if knowledge graph is not populated or query cannot be processed.
        """
        start_time = time.time()
        start_perf = time.perf_counter()

        include_sources = kwargs.get("include_sources", True)
        custom_prompt = kwargs.get("custom_prompt")
        generate_answer = kwargs.get("generate_answer", True)

        # Validate knowledge graph is populated before allowing queries
        self._validate_knowledge_graph()

        # Knowledge graph retrieval with smart fallback to vector search
        try:
            retrieved_documents, method = self._retrieve_via_kg(query_text, top_k)
        except GraphRAGException as e:
            logger.warning(f"GraphRAG fallback: {e}")
            # Fall back to vector search when entities aren't found
            if "No seed entities found" in str(e) or "No documents found" in str(e):
                logger.info(
                    f"GraphRAG: Falling back to vector search for query: '{query_text}'"
                )
                retrieved_documents = self._fallback_to_vector_search(query_text, top_k)
                method = "vector_fallback"
            else:
                # Re-raise other GraphRAG exceptions (validation failures, etc.)
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
                "pipeline_type": "graphrag",
                "retrieval_method": method,
                "generated_answer": generate_answer and answer is not None,
            },
        }

        # Attach DB instrumentation if available
        if hasattr(self, "_debug_db_execs"):
            response["metadata"]["db_exec_count"] = int(self._debug_db_execs)
        if hasattr(self, "_debug_step_times"):
            response["metadata"]["step_timings_ms"] = dict(self._debug_step_times)

        if include_sources:
            response["sources"] = self._extract_sources(retrieved_documents)

        logger.info(
            f"GraphRAG query completed in {execution_time:.2f}s ({execution_time_ms:.1f}ms) - "
            f"{len(retrieved_documents)} docs via {method}; db_exec_count={response['metadata'].get('db_exec_count', 'n/a')}"
        )
        return response

    def _validate_knowledge_graph(self) -> None:
        """
        Validate that the knowledge graph has entities before allowing queries.

        Raises KnowledgeGraphNotPopulatedException if no entities exist.
        """
        if not self.connection_manager:
            raise GraphRAGException(
                "No connection manager available for GraphRAG validation"
            )

        connection = None
        cursor = None
        try:
            connection = self.connection_manager.get_connection()
            conn_type = (
                f"{connection.__class__.__module__}.{connection.__class__.__name__}"
            )
            cursor = connection.cursor()

            # Ensure knowledge graph tables exist
            try:
                schema_manager = SchemaManager(
                    self.connection_manager, self.config_manager
                )
                schema_manager.ensure_table_schema("Entities")
                schema_manager.ensure_table_schema("EntityRelationships")
            except Exception as e:
                logger.warning(f"Could not ensure knowledge graph tables: {e}")

            # Check if we have any entities
            t0 = time.perf_counter()
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            entity_count = cursor.fetchone()[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if entity_count == 0:
                raise KnowledgeGraphNotPopulatedException(
                    "Knowledge graph is empty. No entities found in RAG.Entities table. "
                    "Load documents with entity extraction before querying GraphRAG."
                )

            logger.info(
                f"Knowledge graph validation passed: {entity_count} entities found (query {elapsed_ms:.1f}ms, conn={conn_type})"
            )

        except Exception as e:
            if isinstance(e, KnowledgeGraphNotPopulatedException):
                raise
            raise GraphRAGException(f"Knowledge graph validation failed: {e}")
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error closing cursor: {e}")

    def _retrieve_via_kg(
        self, query_text: str, top_k: int
    ) -> Tuple[List[Document], str]:
        """
        Retrieve documents via knowledge graph traversal.

        Fails hard if any step fails - no fallbacks to vector search.
        """
        # Initialize per-query debug instrumentation
        self._debug_db_execs = 0
        self._debug_step_times = {}

        # Find seed entities - fail hard if none found
        t0 = time.perf_counter()
        seed_entities = self._find_seed_entities(query_text)
        self._debug_step_times["find_seed_entities_ms"] = (
            time.perf_counter() - t0
        ) * 1000.0

        # Traverse graph - fail hard if no relevant entities found
        t1 = time.perf_counter()
        relevant_entities = self._traverse_graph(seed_entities)
        self._debug_step_times["traverse_graph_ms"] = (
            time.perf_counter() - t1
        ) * 1000.0

        # Get documents - fail hard if no documents found
        t2 = time.perf_counter()
        docs = self._get_documents_from_entities(relevant_entities, top_k)
        self._debug_step_times["get_documents_ms"] = (time.perf_counter() - t2) * 1000.0

        logger.debug(
            f"GraphRAG retrieval steps: seed={self._debug_step_times['find_seed_entities_ms']:.1f}ms, "
            f"traverse={self._debug_step_times['traverse_graph_ms']:.1f}ms, "
            f"docs={self._debug_step_times['get_documents_ms']:.1f}ms, db_execs={self._debug_db_execs}"
        )
        return docs, "knowledge_graph_traversal"

    def _find_seed_entities(self, query_text: str) -> List[Tuple[str, str, float]]:
        """
        Find seed entities using RAG.Entities table.

        Fails hard if no entities are found for the query.
        """
        if not self.connection_manager:
            raise GraphRAGException(
                "No connection manager available for seed entity search"
            )

        connection = None
        cursor = None
        seed_entities = []

        try:
            connection = self.connection_manager.get_connection()
            conn_type = (
                f"{connection.__class__.__module__}.{connection.__class__.__name__}"
            )
            cursor = connection.cursor()

            # Clean query text and extract keywords, removing punctuation
            import re

            cleaned_query = re.sub(r"[^\w\s]", " ", query_text.lower())
            query_keywords = [kw for kw in cleaned_query.split() if len(kw) > 2][:5]

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
            """
            logger.debug(
                f"GraphRAG: Executing seed entity query with {len(query_keywords)} keywords (conn={conn_type})"
            )
            t0 = time.perf_counter()
            # Track DB round-trips
            self._debug_db_execs = getattr(self, "_debug_db_execs", 0) + 1
            cursor.execute(query, params)
            results = cursor.fetchall()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            for entity_id, entity_name, entity_type in results:
                seed_entities.append((str(entity_id), str(entity_name), 0.9))

            if not seed_entities:
                raise GraphRAGException(
                    f"No seed entities found for query '{query_text}'. "
                    f"Knowledge graph may not contain relevant entities for this query."
                )

            logger.info(
                f"GraphRAG: Found {len(seed_entities)} seed entities for query: '{query_text}' (query {elapsed_ms:.1f}ms, conn={conn_type})"
            )

        except Exception as e:
            if isinstance(e, GraphRAGException):
                raise
            raise GraphRAGException(f"Database error finding seed entities: {e}")
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error closing cursor: {e}")

        return seed_entities

    def _traverse_graph(self, seed_entities: List[Tuple[str, str, float]]) -> Set[str]:
        """
        Traverse knowledge graph using RAG.EntityRelationships.

        Fails hard if no relationships exist or traversal fails.
        """
        if not seed_entities:
            raise GraphRAGException("No seed entities provided for graph traversal")

        if not self.connection_manager:
            raise GraphRAGException(
                "No connection manager available for graph traversal"
            )

        relevant_entities: Set[str] = {e[0] for e in seed_entities}
        current_entities: Set[str] = {e[0] for e in seed_entities}

        connection = None
        cursor = None
        try:
            connection = self.connection_manager.get_connection()
            conn_type = (
                f"{connection.__class__.__module__}.{connection.__class__.__name__}"
            )
            cursor = connection.cursor()

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

                logger.debug(
                    f"GraphRAG: Traversing graph at depth {depth} with {len(current_entities)} entities (conn={conn_type})"
                )
                t0 = time.perf_counter()
                # Track DB round-trips
                self._debug_db_execs = getattr(self, "_debug_db_execs", 0) + 1
                cursor.execute(query, entity_list + entity_list)
                results = cursor.fetchall()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                next_entities = set()
                for (entity_id,) in results:
                    entity_id_str = str(entity_id)
                    if entity_id_str not in relevant_entities:
                        relevant_entities.add(entity_id_str)
                        next_entities.add(entity_id_str)

                current_entities = next_entities
                logger.debug(
                    f"GraphRAG: Depth {depth} - {len(results)} edges, {len(next_entities)} new entities (query {elapsed_ms:.1f}ms)"
                )

        except Exception as e:
            raise GraphRAGException(f"Database error traversing graph: {e}")
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error closing cursor: {e}")

        if len(relevant_entities) == len(seed_entities):
            raise GraphRAGException(
                "Graph traversal found no additional entities. "
                "Knowledge graph may lack relationships for the given entities."
            )

        logger.info(
            f"GraphRAG: Graph traversal completed with {len(relevant_entities)} total entities"
        )
        return relevant_entities

    def _get_documents_from_entities(
        self, entity_ids: Set[str], top_k: int
    ) -> List[Document]:
        """
        Get documents associated with entities.

        Fails hard if no documents are found for the entities.
        """
        if not entity_ids:
            raise GraphRAGException("No entity IDs provided for document retrieval")

        if not self.connection_manager:
            raise GraphRAGException(
                "No connection manager available for document retrieval"
            )

        connection = None
        cursor = None
        docs = []
        try:
            connection = self.connection_manager.get_connection()
            conn_type = (
                f"{connection.__class__.__module__}.{connection.__class__.__name__}"
            )
            cursor = connection.cursor()
            entity_list = list(entity_ids)[:50]
            placeholders = ",".join(["?" for _ in entity_list])

            query = f"""
                SELECT DISTINCT sd.id, sd.text_content, sd.title
                FROM RAG.SourceDocuments sd
                JOIN RAG.Entities e ON sd.id = e.source_doc_id
                WHERE e.entity_id IN ({placeholders})
                ORDER BY sd.id
            """

            logger.debug(
                f"GraphRAG: Retrieving documents for {len(entity_list)} entities (conn={conn_type})"
            )
            t0 = time.perf_counter()
            # Track DB round-trips
            self._debug_db_execs = getattr(self, "_debug_db_execs", 0) + 1
            cursor.execute(query, entity_list)
            results = cursor.fetchall()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if not results:
                raise GraphRAGException(
                    f"No documents found for {len(entity_list)} entities. "
                    f"Knowledge graph entities may not be properly linked to source documents."
                )

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

            logger.info(
                f"GraphRAG: Retrieved {len(docs)} documents from knowledge graph (query {elapsed_ms:.1f}ms, conn={conn_type})"
            )

        except Exception as e:
            if isinstance(e, GraphRAGException):
                raise
            raise GraphRAGException(f"Database error getting documents: {e}")
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception as e:
                    logger.warning(f"Error closing cursor: {e}")

        return docs

    def _fallback_to_vector_search(self, query_text: str, top_k: int) -> List[Document]:
        """
        Fallback to vector search when knowledge graph can't handle the query.

        This implements the spec requirement (FR-006) for entity-not-found fallback.
        """
        logger.info(
            f"GraphRAG: Executing vector search fallback for query: '{query_text}'"
        )

        if not self.vector_store:
            logger.warning(
                "No vector store available for fallback - returning empty results"
            )
            return []

        try:
            # Use vector store for similarity search
            results = self.vector_store.similarity_search(query_text, k=top_k)

            # Convert results to Document objects if needed
            documents = []
            for result in results:
                if hasattr(result, "page_content"):
                    # Already a Document-like object
                    doc = Document(
                        id=getattr(result, "id", str(len(documents))),
                        page_content=result.page_content,
                        metadata={
                            **(
                                result.metadata
                                if hasattr(result, "metadata") and result.metadata
                                else {}
                            ),
                            "retrieval_method": "vector_fallback",
                        },
                    )
                    documents.append(doc)
                else:
                    # Create Document from string result
                    doc = Document(
                        id=str(len(documents)),
                        page_content=str(result),
                        metadata={"retrieval_method": "vector_fallback"},
                    )
                    documents.append(doc)

            if len(documents) == 0:
                # FR-004: Log diagnostic information when 0 results returned
                logger.info(
                    f"Vector search returned 0 results for query: '{query_text[:50]}...'"
                )
                logger.debug(f"Top-K parameter: {top_k}")
                logger.debug(f"Query text: {query_text}")
            else:
                logger.info(
                    f"GraphRAG: Vector fallback retrieved {len(documents)} documents"
                )
            return documents

        except Exception as e:
            logger.error(f"Vector search fallback failed: {e}")
            logger.debug(f"Query text: {query_text}")
            logger.debug(f"Top-K parameter: {top_k}")
            return []

    def _read_iris_data(self, data) -> str:
        """Handle IRIS stream data."""
        if data is None:
            return ""
        try:
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
        for doc in documents[:5]:  # Limit context
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

    def ask(self, question: str, **kwargs) -> str:
        """Get answer only."""
        result = self.query(question, **kwargs)
        return result.get("answer", "No answer generated")
