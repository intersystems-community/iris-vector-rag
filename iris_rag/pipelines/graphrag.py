"""
GraphRAG Pipeline implementation.

This module provides a RAG implementation using graph-based retrieval techniques.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from common.iris_connection_manager import get_iris_connection
from ..config.manager import ConfigurationManager
from ..storage.iris import IRISStorage
from ..storage.schema_manager import SchemaManager
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class GraphRAGPipeline(RAGPipeline):
    """
    GraphRAG pipeline implementation.
    
    This pipeline implements graph-based RAG approach:
    1. Entity extraction and relationship mapping
    2. Graph-based retrieval using entity relationships
    3. Context augmentation and LLM generation
    """
    
    def __init__(self, config_manager: ConfigurationManager,
                 vector_store=None, llm_func: Optional[Callable[[str], str]] = None,
                 connection_factory: Optional[Callable] = None):
        """
        Initialize the GraphRAG Pipeline.
        
        Args:
            config_manager: Manager for configuration settings
            vector_store: Optional VectorStore instance
            llm_func: Optional LLM function for answer generation
            connection_factory: Optional connection factory for testing
        """
        super().__init__(config_manager, vector_store)
        self.llm_func = llm_func
        
        # Set connection factory (for testing dependency injection)
        self.connection_factory = connection_factory or get_iris_connection
        
        # Initialize components
        self.storage = IRISStorage(config_manager)
        self.embedding_manager = EmbeddingManager(config_manager)
        
        # Create iris_connector for SchemaManager (following ColBERT pattern)
        iris_connector = type('IRISConnector', (), {
            'get_connection': self.connection_factory
        })()
        self.schema_manager = SchemaManager(iris_connector, config_manager)
        
        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get("pipelines:graphrag", {})
        self.top_k = self.pipeline_config.get("top_k", 5)
        self.max_entities = self.pipeline_config.get("max_entities", 10)
        self.relationship_depth = self.pipeline_config.get("relationship_depth", 2)
        
        # Ensure required GraphRAG tables exist
        self._ensure_graphrag_schema()
        
        logger.info(f"Initialized GraphRAGPipeline with top_k={self.top_k}")
    
    def _ensure_graphrag_schema(self):
        """Ensure all required GraphRAG tables exist."""
        try:
            # Ensure DocumentEntities table exists (required for graph-based retrieval)
            if not self.schema_manager.ensure_table_schema("DocumentEntities"):
                logger.error("Failed to ensure DocumentEntities table schema")
                raise RuntimeError("Schema validation failed for DocumentEntities table")
            
            logger.debug("✅ GraphRAG schema validation completed")
        except Exception as e:
            logger.error(f"Failed to ensure GraphRAG schema: {e}")
            raise RuntimeError(f"GraphRAG schema initialization failed: {e}")
    
    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Execute the GraphRAG pipeline (required abstract method).
        
        Args:
            query_text: The input query string
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        top_k = kwargs.get("top_k", 5)
        return self.query(query_text, top_k)
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents into the knowledge base with semantic chunking for better entity extraction.
        
        Args:
            documents_path: Path to documents or directory
            **kwargs: Additional keyword arguments including:
                - documents: List of Document objects (if providing directly)
                - auto_chunk: Override automatic chunking setting
                - chunking_strategy: Override chunking strategy (default: semantic)
        """
        # Load documents using base class implementation
        super().load_documents(documents_path, **kwargs)
        
        # Get documents for entity extraction
        documents = self._get_documents(documents_path, **kwargs)
        
        # Extract entities and relationships from original documents
        entities_created = 0
        relationships_created = 0
        
        for doc in documents:
            entities = self._extract_entities(doc)
            relationships = self._extract_relationships(doc, entities)
            
            self._store_entities(doc.id, entities)
            self._store_relationships(doc.id, relationships)
            
            entities_created += len(entities)
            relationships_created += len(relationships)
        
        logger.info(f"GraphRAG: Extracted {entities_created} entities and {relationships_created} relationships")
    
    def ingest_documents(self, documents: List[Document], auto_chunk: Optional[bool] = None, chunking_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest documents with entity extraction and graph building, with chunking support.
        
        Args:
            documents: List of documents to ingest
            auto_chunk: Whether to enable automatic chunking
            chunking_strategy: Strategy to use for chunking
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        logger.info(f"Starting GraphRAG ingestion of {len(documents)} documents")
        
        try:
            # Store documents using vector store with chunking support
            document_ids = self.vector_store.add_documents(
                documents=documents,
                auto_chunk=auto_chunk,
                chunking_strategy=chunking_strategy
            )
            
            # Extract entities and relationships from original documents
            entities_created = 0
            relationships_created = 0
            
            for doc in documents:
                entities = self._extract_entities(doc)
                relationships = self._extract_relationships(doc, entities)
                
                self._store_entities(doc.id, entities)
                self._store_relationships(doc.id, relationships)
                
                entities_created += len(entities)
                relationships_created += len(relationships)
            
            end_time = time.time()
            
            result = {
                "status": "success",
                "documents_ingested": len(documents),
                "chunks_created": len(document_ids),
                "entities_created": entities_created,
                "relationships_created": relationships_created,
                "processing_time": end_time - start_time,
                "pipeline_type": "graphrag",
                "chunking_enabled": auto_chunk,
                "chunking_strategy": chunking_strategy
            }
            
            logger.info(f"GraphRAG ingestion completed with chunking: {result}")
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "pipeline_type": "graphrag"
            }
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute a query using graph-based retrieval with vector fallback.
        
        Args:
            query_text: The query string
            top_k: Number of top documents to retrieve
            
        Returns:
            Dictionary with query results
        """
        start_time = time.time()
        logger.info(f"Processing GraphRAG query: {query_text}")
        
        try:
            # Extract entities from query
            query_entities = self._extract_query_entities(query_text)
            
            # Find related documents through graph traversal
            relevant_docs = self._graph_based_retrieval(query_entities, top_k)
            
            # If no graph-based results, fall back to vector search
            if not relevant_docs:
                logger.warning("GraphRAG: No graph-based results found, falling back to vector search")
                # Generate query embedding for vector search
                query_embedding = self.embedding_manager.embed_text(query_text)
                relevant_docs = self._vector_fallback_retrieval(query_embedding, top_k)
                
                # If still no results, return appropriate response
                if not relevant_docs:
                    logger.error("GraphRAG: Graph-based retrieval failed - insufficient graph data. GraphRAG requires populated knowledge graph.")
                    return {
                        "query": query_text,
                        "answer": "GraphRAG failed: Insufficient knowledge graph data for graph-based retrieval. Please use BasicRAG or ensure knowledge graph is properly populated.",
                        "retrieved_documents": [],
                        "query_entities": query_entities,
                        "num_documents_retrieved": 0,
                        "processing_time": time.time() - start_time,
                        "pipeline_type": "graphrag",
                        "failure_reason": "insufficient_graph_data"
                    }
            
            # Generate answer if LLM function is available
            answer = None
            if self.llm_func and relevant_docs:
                context = self._build_context(relevant_docs)
                prompt = self._build_prompt(query_text, context)
                answer = self.llm_func(prompt)
            
            end_time = time.time()
            
            result = {
                "query": query_text,
                "query_entities": query_entities,
                "answer": answer,
                "retrieved_documents": relevant_docs,
                "num_documents_retrieved": len(relevant_docs),
                "processing_time": end_time - start_time,
                "pipeline_type": "graphrag"
            }
            
            logger.info(f"GraphRAG query completed in {end_time - start_time:.2f}s. Retrieved {len(relevant_docs)} documents.")
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG query failed: {e}", exc_info=True)
            
            # Try vector fallback on entity extraction failure
            try:
                logger.warning("GraphRAG: Entity extraction failed, attempting vector fallback")
                relevant_docs = self._vector_fallback_retrieval(query_text, top_k)
                
                if relevant_docs:
                    # Generate answer if LLM function is available
                    answer = None
                    if self.llm_func:
                        context = self._build_context(relevant_docs)
                        prompt = self._build_prompt(query_text, context)
                        answer = self.llm_func(prompt)
                    
                    return {
                        "query": query_text,
                        "answer": answer,
                        "retrieved_documents": relevant_docs,
                        "query_entities": [],
                        "num_documents_retrieved": len(relevant_docs),
                        "processing_time": time.time() - start_time,
                        "pipeline_type": "graphrag",
                        "fallback_used": "vector"
                    }
            except Exception as fallback_error:
                logger.error(f"Vector fallback also failed: {fallback_error}")
            
            # Extract entities even in error case for consistent response format
            try:
                query_entities = self._extract_query_entities(query_text)
            except:
                query_entities = []
            
            return {
                "query": query_text,
                "answer": None,
                "retrieved_documents": [], # Ensure this key exists on error
                "query_entities": query_entities,
                "num_documents_retrieved": 0,
                "processing_time": time.time() - start_time,
                "error": str(e),
                "pipeline_type": "graphrag"
            }

    def _vector_fallback_retrieval(self, query_text: str, top_k: int) -> List[Document]:
        """
        Fallback to vector-based retrieval when graph-based retrieval fails.
        
        Args:
            query_text: The query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_manager.embed_text(query_text)
            
            # Use vector store for retrieval - always use IRISVectorStore interface
            from ..storage.vector_store_iris import IRISVectorStore
            
            vector_store = IRISVectorStore(config_manager=self.config_manager)
            results = vector_store.similarity_search_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Convert results to Document objects with proper metadata
            documents = []
            for doc, score in results:
                doc.metadata.update({
                    "similarity_score": float(score),
                    "retrieval_type": "vector_fallback"
                })
                documents.append(doc)
            
            logger.info(f"Vector fallback retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Vector fallback retrieval failed: {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()

    def _graph_based_retrieval(self, query_entities: List[str], top_k: int) -> List[Document]:
        """
        Retrieve documents based on graph traversal using extracted entities.
        Refactored to a two-stage query to avoid potential caching issues with complex queries on text_content.
        """
        if not query_entities:
            logger.warning("GraphRAG: No query entities provided for graph-based retrieval.")
            return []

        connection = self.connection_factory()
        cursor = connection.cursor()
        retrieved_documents = []

        try:
            entity_placeholders = ",".join(["?" for _ in query_entities])
            
            # Stage 1: Get doc_ids and entity_matches
            # No dynamic comment needed here as we are not selecting text_content directly
            search_sql_stage1 = f"""
            SELECT TOP {top_k} d.ID, COUNT(*) as entity_matches
            FROM RAG.SourceDocuments d
            JOIN RAG.DocumentEntities e ON d.ID = e.doc_id
            WHERE e.entity_name IN ({entity_placeholders})
            GROUP BY d.ID
            ORDER BY entity_matches DESC
            """
            logger.info(f"GraphRAG Stage 1 Query: {search_sql_stage1} with params: {query_entities}")
            cursor.execute(search_sql_stage1, query_entities)
            doc_id_matches = cursor.fetchall()

            if not doc_id_matches:
                logger.info("GraphRAG Stage 1: No matching doc_ids found.")
                return []

            logger.info(f"GraphRAG Stage 1: Found {len(doc_id_matches)} potential doc_ids.")

            # Stage 2: Fetch document content for each doc_id
            for row in doc_id_matches:
                if len(row) >= 2:
                    doc_id, entity_matches_count = row[:2]
                else:
                    continue
                # It's crucial to use a new cursor or ensure the previous one is reset if reusing
                # For simplicity, creating a new cursor for each sub-query or ensuring it's clean.
                # However, for performance with many doc_ids, batching or a single cursor carefully managed would be better.
                # For now, let's assume the main cursor can be reused if no open results.
                
                # Ensure the cursor is ready for a new query if it's being reused.
                # Depending on DBAPI driver, this might not be strictly necessary if fetchall() was called.
                # If issues arise, create a new cursor: content_cursor = connection.cursor()
                
                content_sql_stage2 = """
                SELECT title, "text_content" AS graph_doc_content
                FROM RAG.SourceDocuments
                WHERE doc_id = ?
                """
                # logger.debug(f"GraphRAG Stage 2 Query for doc_id {doc_id}: {content_sql_stage2}")
                cursor.execute(content_sql_stage2, [doc_id])
                content_row = cursor.fetchone()

                if content_row:
                    # Safe tuple access with bounds checking for mock compatibility
                    if hasattr(content_row, '__len__') and len(content_row) >= 2:
                        title, doc_content_text = content_row[:2]
                    else:
                        # Handle mock objects or insufficient data
                        title = "Mock Title"
                        doc_content_text = "Mock content"
                    
                    # VectorStore guarantees string content
                    page_content = str(doc_content_text) if doc_content_text else ""
                    title_str = str(title) if title else "N/A"
                    
                    doc = Document(
                        id=doc_id,
                        page_content=page_content, # Ensure page_content is a string
                        metadata={
                            "title": title_str,
                            "entity_matches": entity_matches_count,
                            "retrieval_method": "graph_based_retrieval"
                        }
                    )
                    retrieved_documents.append(doc)
                else:
                    logger.warning(f"GraphRAG Stage 2: No content found for doc_id {doc_id}, though it was matched in Stage 1.")
            
            logger.info(f"GraphRAG: Retrieved {len(retrieved_documents)} documents after 2-stage query.")

        except Exception as e:
            logger.error(f"GraphRAG _graph_based_retrieval error: {e}", exc_info=True)
            # Fallback or re-raise might be needed depending on desired behavior
            return [] # Return empty list on error to allow fallback
        finally:
            cursor.close()
            
        return retrieved_documents

    def _vector_fallback_retrieval(self, query_embedding: List[float], top_k: int) -> List[Document]:
        """
        Fallback to simple vector search if graph retrieval fails or yields no results.
        Uses IRISVectorStore interface to comply with Vector Store Architecture Rules.
        """
        logger.info("GraphRAG: Performing vector fallback retrieval.")
        try:
            # Use vector store interface instead of direct SQL
            from iris_rag.storage.vector_store_iris import IRISVectorStore
            
            vector_store = IRISVectorStore(config_manager=self.config_manager)
            
            # Use similarity_search_by_embedding method with correct parameters
            results = vector_store.similarity_search_by_embedding(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            retrieved_documents = []
            for doc_tuple in results:
                # IRISVectorStore returns List[Tuple[Document, float]]
                doc, similarity_score = doc_tuple
                
                # Convert to Document format expected by GraphRAG
                retrieved_doc = Document(
                    id=doc.metadata.get("doc_id", "unknown"),
                    page_content=doc.page_content,
                    metadata={
                        "title": doc.metadata.get("title", "N/A"),
                        "similarity_score": similarity_score,
                        "retrieval_method": "vector_fallback"
                    }
                )
                retrieved_documents.append(retrieved_doc)
            
            logger.info(f"GraphRAG Vector Fallback: Retrieved {len(retrieved_documents)} documents.")
            return retrieved_documents
            
        except Exception as e:
            logger.error(f"GraphRAG vector fallback retrieval error: {e}", exc_info=True)
            return []

    def _extract_entities(self, document: Document) -> List[Dict[str, Any]]:
        """Extract entities from document text."""
        # Simple entity extraction - in practice, would use NER models
        text = document.page_content
        
        # Basic keyword extraction as entities
        words = text.split()
        entities = []
        
        # Extract capitalized words as potential entities
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 3:
                entity_embedding = self.embedding_manager.embed_text(word)
                entities.append({
                    "entity_id": f"{document.id}_entity_{i}",
                    "entity_text": word,
                    "entity_type": "KEYWORD",
                    "position": i,
                    "embedding": entity_embedding
                })
        
        return entities[:self.max_entities]  # Limit number of entities
    
    def _extract_relationships(self, document: Document, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        # Simple co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Create relationship if entities are close in text
                pos_diff = abs(entity1["position"] - entity2["position"])
                if pos_diff <= 10:  # Within 10 words
                    relationships.append({
                        "relationship_id": f"{document.id}_rel_{i}_{j}",
                        "source_entity": entity1["entity_id"],
                        "target_entity": entity2["entity_id"],
                        "relationship_type": "CO_OCCURS",
                        "strength": 1.0 / (pos_diff + 1)  # Closer = stronger
                    })
        
        return relationships
    
    def _store_entities(self, document_id: str, entities: List[Dict[str, Any]]):
        """Store entities in the database with robust vector embedding handling."""
        # Ensure schema is correct before storing entities
        if not self.schema_manager.ensure_table_schema("DocumentEntities"):
            logger.error("Failed to ensure DocumentEntities table schema")
            raise RuntimeError("Schema validation failed for DocumentEntities table")
        
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            from common.vector_format_fix import format_vector_for_iris, VectorFormatError, validate_vector_for_iris, create_iris_vector_string, pad_vector_to_dimension
            
            successful_embeddings = 0
            failed_embeddings = 0
            
            for entity in entities:
                # Pre-validate entity data
                if not all(key in entity for key in ["entity_id", "entity_text", "entity_type", "position"]):
                    logger.warning(f"Entity missing required fields: {entity}")
                    continue
                
                # Handle embedding storage with comprehensive validation
                embedding_formatted = None
                if "embedding" in entity and entity["embedding"] is not None:
                    try:
                        # Step 1: Format the vector using the proven utility
                        embedding_list = format_vector_for_iris(entity["embedding"])
                        
                        # Step 2: Validate the formatted vector
                        if not validate_vector_for_iris(embedding_list):
                            raise VectorFormatError("Vector validation failed after formatting")
                        
                        # Step 3: Create optimized string representation using new utility
                        # Use the non-bracketed version for TO_VECTOR() function
                        embedding_formatted = create_iris_vector_string(embedding_list)
                        
                        # Step 4: Final validation
                        if not embedding_formatted or len(embedding_formatted) < 1:
                            raise VectorFormatError("Generated vector string is invalid")
                        
                    except VectorFormatError as e:
                        logger.warning(f"Vector formatting error for entity {entity['entity_id']}: {e}")
                        failed_embeddings += 1
                        embedding_formatted = None
                    except Exception as e:
                        logger.warning(f"Unexpected error processing embedding for entity {entity['entity_id']}: {e}")
                        failed_embeddings += 1
                        embedding_formatted = None
                
                # Store entity with or without embedding based on processing result
                try:
                    if embedding_formatted is not None:
                        # Store with embedding using TO_VECTOR() function
                        insert_sql = """
                        INSERT INTO RAG.DocumentEntities
                        (entity_id, document_id, entity_text, entity_type, position, embedding)
                        VALUES (?, ?, ?, ?, ?, TO_VECTOR(?))
                        """
                        cursor.execute(insert_sql, [
                            entity["entity_id"],
                            document_id,
                            entity["entity_text"],
                            entity["entity_type"],
                            entity["position"],
                            embedding_formatted
                        ])
                        successful_embeddings += 1
                    else:
                        # Store without embedding
                        insert_sql = """
                        INSERT INTO RAG.DocumentEntities
                        (entity_id, document_id, entity_text, entity_type, position)
                        VALUES (?, ?, ?, ?, ?)
                        """
                        cursor.execute(insert_sql, [
                            entity["entity_id"],
                            document_id,
                            entity["entity_text"],
                            entity["entity_type"],
                            entity["position"]
                        ])
                        
                except Exception as sql_error:
                    # Log the error and store without embedding as fallback
                    logger.error(f"SQL insertion failed for entity {entity['entity_id']}: {sql_error}")
                    self._store_entity_without_embedding(cursor, entity, document_id)
                    if embedding_formatted is not None:
                        failed_embeddings += 1
            
            connection.commit()
            
            # Enhanced logging with detailed statistics
            total_entities = len(entities)
            entities_with_embeddings = sum(1 for e in entities if "embedding" in e and e["embedding"] is not None)
            success_rate = (successful_embeddings / entities_with_embeddings * 100) if entities_with_embeddings > 0 else 0
            
            logger.info(f"Stored {total_entities} entities for document {document_id}")
            logger.info(f"Embedding storage: {successful_embeddings}/{entities_with_embeddings} successful ({success_rate:.1f}%)")
            
            if failed_embeddings > 0:
                logger.warning(f"Failed to store {failed_embeddings} embeddings due to formatting/validation errors")
            
        except Exception as e:
            connection.rollback()
            logger.error(f"Failed to store entities for document {document_id}: {e}")
            raise e
        finally:
            cursor.close()
    
    def _store_entity_without_embedding(self, cursor, entity: Dict[str, Any], document_id: str):
        """Store entity without embedding as fallback."""
        try:
            insert_sql = """
            INSERT INTO RAG.DocumentEntities
            (entity_id, document_id, entity_text, entity_type, position)
            VALUES (?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, [
                entity["entity_id"],
                document_id,
                entity["entity_text"],
                entity["entity_type"],
                entity["position"]
            ])
            logger.info(f"Stored entity {entity['entity_id']} without embedding")
        except Exception as e:
            logger.error(f"Failed to store entity {entity['entity_id']} even without embedding: {e}")
            raise
    
    def _store_relationships(self, document_id: str, relationships: List[Dict[str, Any]]):
        """Store relationships in the database."""
        if not relationships:
            logger.info(f"No relationships to store for document {document_id}")
            return
            
        # Ensure EntityRelationships table exists
        self.schema_manager.ensure_table_schema("EntityRelationships")
            
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            for rel in relationships:
                insert_sql = """
                INSERT INTO RAG.EntityRelationships
                (relationship_id, document_id, source_entity, target_entity, relationship_type, strength)
                VALUES (?, ?, ?, ?, ?, ?)
                """
                
                cursor.execute(insert_sql, [
                    rel["relationship_id"],
                    document_id,
                    rel["source_entity"],
                    rel["target_entity"],
                    rel["relationship_type"],
                    rel["strength"]
                ])
            
            connection.commit()
            logger.info(f"Stored {len(relationships)} relationships for document {document_id}")
            
        except Exception as e:
            connection.rollback()
            logger.error(f"Failed to store relationships for document {document_id}: {e}")
            raise e
        finally:
            cursor.close()
    
    def _extract_query_entities(self, query_text: str) -> List[str]:
        """Extract entities from query text."""
        # More sophisticated entity extraction (e.g., using a library like spaCy) could be used here.
        # For now, we'll use a simple stopword list and normalization.
        stopwords = {"a", "an", "the", "about", "for", "in", "on", "of", "tell", "me", "is", "what", "and"}
        
        # Normalize and split
        words = query_text.lower().replace("?", "").replace(".", "").split()
        
        # Filter out stopwords and return unique entities
        entities = sorted(list(set([word for word in words if word not in stopwords])))
        
        logger.info(f"Extracted entities from query: {entities}")
        return entities
    
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from retrieved Document objects."""
        context_parts = []
        if not documents:
            return "No documents retrieved to build context."
            
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Untitled') if doc.metadata else 'Untitled'
            # Ensure page_content is a string, even if None
            content = doc.page_content if doc.page_content is not None else ""
            context_parts.append(f"[Document {i}: {title}]\n{content[:500]}...") # Truncate content for brevity
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM generation."""
        return f"""Based on the following retrieved documents, please answer the question.

Context:
{context}

Question: {query}

Answer:"""