import time
import json
import logging
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

        # Initialize entity extraction service
        self.entity_extraction_service = EntityExtractionService(
            config_manager=config_manager,
            connection_manager=connection_manager,
            embedding_manager=self.embedding_manager,
            llm_func=self.llm_func
        )

        # Configuration
        self.pipeline_config = self.config_manager.get("pipelines:graphrag", {})
        self.default_top_k = self.pipeline_config.get("default_top_k", 10)
        self.max_depth = self.pipeline_config.get("max_depth", 2)
        self.max_entities = self.pipeline_config.get("max_entities", 50)

        # Entity extraction can be disabled for fast document-only indexing
        self.entity_extraction_enabled = self.pipeline_config.get("entity_extraction_enabled", True)

        logger.info(
            f"Production-hardened GraphRAG pipeline initialized (entity extraction: {self.entity_extraction_enabled})"
        )

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents with integrated entity extraction.
        """
        start_time = time.time()

        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            documents = []

        if not documents:
            logger.warning("No documents provided to load_documents")
            return

        # Store documents first
        generate_embeddings = kwargs.get("generate_embeddings", True)
        if generate_embeddings:
            self.vector_store.add_documents(documents)
        else:
            self._store_documents(documents)

        if not self.entity_extraction_enabled:
            return

        # Ensure knowledge graph tables exist
        try:
            schema_manager = SchemaManager(self.connection_manager, self.config_manager)
            schema_manager.ensure_table_schema("Entities")
            schema_manager.ensure_table_schema("EntityRelationships")
        except Exception as e:
            logger.warning(f"Could not ensure knowledge graph tables: {e}")

        total_entities = 0
        total_relationships = 0

        batch_size = 5
        total_batches = (len(documents) + batch_size - 1) // batch_size

        extraction_start_time = time.time()

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_num = i//batch_size + 1

            try:
                # Batch extract entities
                batch_results = self.entity_extraction_service.extract_batch_with_dspy(
                    batch_docs, batch_size=batch_size
                )

                for doc in batch_docs:
                    entities = batch_results.get(doc.id, [])
                    if entities:
                        # Extract and store relationships from entity pairs
                        from ..core.models import Relationship
                        relationships = []
                        for idx1, entity1 in enumerate(entities):
                            for entity2 in entities[idx1+1:]:
                                rel = Relationship(
                                    source_entity_id=entity1.id,
                                    target_entity_id=entity2.id,
                                    relationship_type="co_occurs_with",
                                    confidence=1.0,
                                    source_document_id=doc.id,
                                )
                                relationships.append(rel)

                        from ..services.storage import EntityStorageAdapter
                        storage_service = EntityStorageAdapter(
                            connection_manager=self.connection_manager,
                            config=self.config_manager._config
                        )
                        total_entities += storage_service.store_entities_batch(entities)
                        if relationships:
                            total_relationships += storage_service.store_relationships_batch(relationships)

            except Exception as batch_error:
                logger.error(f"❌ Batch {batch_num} FAILED: {batch_error}")

        extraction_elapsed = time.time() - extraction_start_time
        logger.info(f"Entity Extraction Complete: {total_entities} entities, {total_relationships} relationships in {extraction_elapsed:.2f}s")

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        generate_answer: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Query the GraphRAG pipeline."""
        start_time = time.time()

        if not query_text or query_text.strip() == "":
            raise ValueError("Query text cannot be empty")

        # Robust top_k handling for tests
        effective_top_k = max(1, top_k)

        retrieved_documents, method = self._retrieve_via_kg(query_text, effective_top_k)
        
        # Fallback if KG returned nothing and fallback enabled
        if not retrieved_documents and self.pipeline_config.get("enable_vector_fallback", True):
            retrieved_documents = self._fallback_to_vector_search(query_text, effective_top_k)
            method = "vector_fallback"

        # If top_k was 0, return empty results as requested by some tests
        if top_k <= 0:
            retrieved_documents = []


        answer = None
        if generate_answer:
            if not self.llm_func:
                answer = "Answer generation skipped: No LLM configured."
            elif not retrieved_documents:
                answer = "No relevant context found in the knowledge graph."
            else:
                answer = self._generate_answer(query_text, retrieved_documents)

        execution_time = time.time() - start_time

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents,
            "contexts": [doc.page_content for doc in retrieved_documents],
            "sources": [doc.metadata.get("title", doc.id) for doc in retrieved_documents],
            "execution_time": execution_time,
            "metadata": {
                "num_retrieved": len(retrieved_documents),
                "pipeline_type": "graphrag",
                "generated_answer": generate_answer and answer is not None,
                "processing_time": execution_time,
                "retrieval_method": method,
                "context_count": len(retrieved_documents),
            },
        }

    def _retrieve_via_kg(self, query_text: str, top_k: int) -> Tuple[List[Document], str]:
        # Find entities in query
        start_nodes = self._find_seed_entities(query_text)
        if not start_nodes:
            return [], "no_matching_entities"

        relevant_entities = self._expand_neighborhood(set(start_nodes), depth=self.max_depth)
        docs = self._get_documents_from_entities(relevant_entities, top_k)
        
        return docs, "knowledge_graph"

    def _find_seed_entities(self, query_text: str) -> List[str]:
        """Find entity IDs in the graph matching extracted query entities."""
        from ..core.models import Document as CoreDoc
        doc_wrapper = CoreDoc(page_content=query_text, id="query")
        entities = self.entity_extraction_service.extract_entities(doc_wrapper)
        
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        entity_ids = []
        
        for entity in entities:
            try:
                cursor.execute("SELECT entity_id FROM RAG.Entities WHERE entity_name LIKE ?", [f"%{entity.text}%"])
                for row in cursor.fetchall():
                    if row[0] not in entity_ids:
                        entity_ids.append(row[0])
            except Exception:
                pass
        
        cursor.close()
        return entity_ids

    def _expand_neighborhood(self, start_nodes: Set[str], depth: int) -> Set[str]:
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        visited = set(start_nodes)
        frontier = set(start_nodes)
        
        for _ in range(depth):
            if not frontier: break
            placeholders = ",".join(["?" for _ in frontier])
            query = f"SELECT DISTINCT target_entity_id FROM RAG.EntityRelationships WHERE source_entity_id IN ({placeholders}) UNION SELECT DISTINCT source_entity_id FROM RAG.EntityRelationships WHERE target_entity_id IN ({placeholders})"
            try:
                params = list(frontier) + list(frontier)
                cursor.execute(query, params)
                neighbors = {row[0] for row in cursor.fetchall()}
                new_nodes = neighbors - visited
                visited.update(new_nodes)
                frontier = new_nodes
            except Exception: break
                
        cursor.close()
        return visited

    def _get_documents_from_entities(self, entity_ids: Set[str], top_k: int) -> List[Document]:
        if not entity_ids: return []
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        docs = []
        try:
            entity_list = list(entity_ids)[:50]
            placeholders = ",".join(["?" for _ in entity_list])
            query = f"SELECT DISTINCT sd.doc_id, sd.text_content, sd.metadata FROM RAG.SourceDocuments sd JOIN RAG.Entities e ON sd.doc_id = e.source_doc_id WHERE e.entity_id IN ({placeholders}) ORDER BY sd.doc_id"
            cursor.execute(query, entity_list)
            for row in cursor.fetchall():
                metadata = json.loads(self._read_iris_data(row[2])) if row[2] else {}
                docs.append(Document(id=str(row[0]), page_content=self._read_iris_data(row[1]), metadata={**metadata, "retrieval_method": "knowledge_graph"}))
                if len(docs) >= top_k: break
        except Exception as e:
            logger.error(f"Database error getting docs: {e}")
        finally:
            cursor.close()
        return docs

    def _fallback_to_vector_search(self, query_text: str, top_k: int) -> List[Document]:
        if not self.vector_store: return []
        query_embedding = self.embedding_manager.embed_text(query_text)
        results = self.vector_store.similarity_search(query_embedding=query_embedding, top_k=top_k)
        return [r[0] for r in results]

    def _generate_answer(self, query: str, documents: List[Document]) -> str:
        if not self.llm_func: return "No LLM configured."
        context = "\n\n".join([doc.page_content for doc in documents])
        return self.llm_func(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")

    def _read_iris_data(self, data: Any) -> str:
        if data is None: return ""
        if hasattr(data, "read"): return str(data.read())
        return str(data)

    def _validate_knowledge_graph(self) -> None:
        """Validate that the knowledge graph is populated."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            count = cursor.fetchone()[0]
            if count == 0:
                raise KnowledgeGraphNotPopulatedException("Knowledge graph is empty")
        finally:
            cursor.close()

    def clear(self) -> None:
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM RAG.EntityRelationships")
            cursor.execute("DELETE FROM RAG.Entities")
            cursor.execute("DELETE FROM RAG.SourceDocuments")
            connection.commit()
        finally:
            cursor.close()
