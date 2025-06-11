"""
GraphRAG Pipeline implementation.

This module provides a RAG implementation using graph-based retrieval techniques.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
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
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager,
                 vector_store=None, llm_func: Optional[Callable[[str], str]] = None):
        """
        Initialize the GraphRAG Pipeline.
        
        Args:
            connection_manager: Manager for database connections
            config_manager: Manager for configuration settings
            vector_store: Optional VectorStore instance
            llm_func: Optional LLM function for answer generation
        """
        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func
        
        # Initialize components
        self.storage = IRISStorage(connection_manager, config_manager)
        self.embedding_manager = EmbeddingManager(config_manager)
        self.schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get("pipelines:graphrag", {})
        self.top_k = self.pipeline_config.get("top_k", 5)
        self.max_entities = self.pipeline_config.get("max_entities", 10)
        self.relationship_depth = self.pipeline_config.get("relationship_depth", 2)
        
        logger.info(f"Initialized GraphRAGPipeline with top_k={self.top_k}")
    
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
        Load documents into the knowledge base (required abstract method).
        
        Args:
            documents_path: Path to documents or directory
            **kwargs: Additional keyword arguments including:
                - documents: List of Document objects (if providing directly)
                - chunk_documents: Whether to chunk documents (default: True)
                - generate_embeddings: Whether to generate embeddings (default: True)
        """
        # Handle direct document input
        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            # Load documents from path - basic implementation
            import os
            documents = []
            
            if os.path.isfile(documents_path):
                with open(documents_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": documents_path}
                )
                documents.append(doc)
            elif os.path.isdir(documents_path):
                for filename in os.listdir(documents_path):
                    file_path = os.path.join(documents_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            doc = Document(
                                page_content=content,
                                metadata={"source": file_path, "filename": filename}
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"Failed to load file {file_path}: {e}")
        
        # Use the ingest_documents method
        result = self.ingest_documents(documents)
        logger.info(f"GraphRAG: Loaded {len(documents)} documents - {result}")
    
    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Ingest documents with entity extraction and graph building.
        
        Args:
            documents: List of documents to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        logger.info(f"Starting GraphRAG ingestion of {len(documents)} documents")
        
        try:
            # Store documents first
            ingestion_result = self.storage.store_documents(documents)
            
            # Extract entities and relationships
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
                "entities_created": entities_created,
                "relationships_created": relationships_created,
                "processing_time": end_time - start_time,
                "pipeline_type": "graphrag"
            }
            
            logger.info(f"GraphRAG ingestion completed: {result}")
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
        Execute a query using graph-based retrieval.
        
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
            
            # If no graph-based results, fallback to vector search
            if not relevant_docs:
                logger.info("GraphRAG: No results from graph-based retrieval, falling back to vector search.")
                query_embedding = self.embedding_manager.embed_text(query_text)
                relevant_docs = self._vector_fallback_retrieval(query_embedding, top_k)
            
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
            return {
                "query": query_text,
                "answer": None,
                "retrieved_documents": [], # Ensure this key exists on error
                "num_documents_retrieved": 0,
                "error": str(e),
                "pipeline_type": "graphrag"
            }

    def _graph_based_retrieval(self, query_entities: List[str], top_k: int) -> List[Document]:
        """
        Retrieve documents based on graph traversal using extracted entities.
        Refactored to a two-stage query to avoid potential caching issues with complex queries on text_content.
        """
        if not query_entities:
            logger.warning("GraphRAG: No query entities provided for graph-based retrieval.")
            return []

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        retrieved_documents = []

        try:
            entity_placeholders = ",".join(["?" for _ in query_entities])
            
            # Stage 1: Get doc_ids and entity_matches
            # No dynamic comment needed here as we are not selecting text_content directly
            search_sql_stage1 = f"""
            SELECT TOP {top_k} d.doc_id, COUNT(*) as entity_matches
            FROM RAG.SourceDocuments d
            JOIN RAG.DocumentEntities e ON d.doc_id = e.document_id
            WHERE e.entity_text IN ({entity_placeholders})
            GROUP BY d.doc_id
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
            for doc_id, entity_matches_count in doc_id_matches:
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
                    title, doc_content_text = content_row
                    
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
        """Fallback to simple vector search if graph retrieval fails or yields no results."""
        logger.info("GraphRAG: Performing vector fallback retrieval.")
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        retrieved_documents = []
        try:
            # Ensure query_embedding is a string representation of a list for TO_VECTOR
            query_embedding_str = str(query_embedding)

            sql = f"""
            SELECT TOP {top_k} doc_id, title, text_content, VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
            FROM RAG.SourceDocuments
            ORDER BY similarity DESC
            """
            cursor.execute(sql, [query_embedding_str])
            results = cursor.fetchall()
            for row in results:
                # VectorStore guarantees string content
                page_content = str(row[2]) if row[2] else ""
                title_str = str(row[1]) if row[1] else "N/A"
                
                doc = Document(
                    id=row[0],
                    page_content=page_content,
                    metadata={
                        "title": title_str,
                        "similarity_score": float(row[3]) if row[3] is not None else 0.0,
                        "retrieval_method": "vector_fallback"
                    }
                )
                retrieved_documents.append(doc)
            logger.info(f"GraphRAG Vector Fallback: Retrieved {len(retrieved_documents)} documents.")
        except Exception as e:
            logger.error(f"GraphRAG vector fallback retrieval error: {e}", exc_info=True)
        finally:
            cursor.close()
        return retrieved_documents

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
        
        connection = self.connection_manager.get_connection()
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
            
        connection = self.connection_manager.get_connection()
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
        words = query_text.split()
        entities = []
        
        for word in words:
            if word[0].isupper() and len(word) > 3:
                entities.append(word)
        
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