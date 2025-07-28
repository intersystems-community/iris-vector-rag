"""
Token Embedding Service for ColBERT RAG Pipeline.

This service provides centralized management of token embeddings including:
- Auto-population of missing token embeddings
- Batch processing for efficiency
- Integration with ColBERT interface
- Error handling and logging
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import time
from dataclasses import dataclass

from ..config.manager import ConfigurationManager
from ..embeddings.colbert_interface import get_colbert_interface_from_config
from ..storage.schema_manager import SchemaManager
from common.db_vector_utils import insert_vector

logger = logging.getLogger(__name__)


@dataclass
class TokenEmbeddingStats:
    """Statistics for token embedding operations."""
    documents_processed: int = 0
    tokens_generated: int = 0
    processing_time: float = 0.0
    errors: int = 0


class TokenEmbeddingService:
    """
    Service for managing ColBERT token embeddings.
    
    Provides centralized functionality for:
    - Auto-populating missing token embeddings
    - Batch processing for efficiency
    - Integration with existing ColBERT interface
    - Proper error handling and logging
    """
    
    def __init__(self, config_manager: ConfigurationManager, connection_manager):
        """
        Initialize token embedding service.
        
        Args:
            config_manager: Configuration manager instance
            connection_manager: Database connection manager
        """
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        
        # Initialize schema manager for dimension management
        self.schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Get ColBERT interface for token embedding generation
        self.colbert_interface = get_colbert_interface_from_config(
            config_manager, connection_manager
        )
        
        # Get token dimension from schema manager (should be 768D for ColBERT)
        self.token_dimension = self.schema_manager.get_colbert_token_dimension()
        
        logger.info(f"TokenEmbeddingService initialized with {self.token_dimension}D token embeddings")
    
    def ensure_token_embeddings_exist(self, doc_ids: Optional[List[str]] = None) -> TokenEmbeddingStats:
        """
        Ensure token embeddings exist for specified documents or all documents.
        
        Args:
            doc_ids: Optional list of document IDs to process. If None, processes all documents.
            
        Returns:
            TokenEmbeddingStats with processing information
        """
        logger.info("Starting token embedding auto-population")
        start_time = time.time()
        stats = TokenEmbeddingStats()
        
        try:
            # Ensure DocumentTokenEmbeddings table exists
            self.schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
            
            # Get documents that need token embeddings
            missing_docs = self._get_documents_missing_token_embeddings(doc_ids)
            
            if not missing_docs:
                logger.info("All documents already have token embeddings")
                return stats
            
            logger.info(f"Found {len(missing_docs)} documents missing token embeddings")
            
            # Process documents in batches for efficiency
            batch_size = self.config_manager.get("colbert.batch_size", 50)
            
            for i in range(0, len(missing_docs), batch_size):
                batch = missing_docs[i:i + batch_size]
                batch_stats = self._process_document_batch(batch)
                
                stats.documents_processed += batch_stats.documents_processed
                stats.tokens_generated += batch_stats.tokens_generated
                stats.errors += batch_stats.errors
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(missing_docs) + batch_size - 1)//batch_size}")
            
            stats.processing_time = time.time() - start_time
            
            logger.info(f"Token embedding auto-population completed: "
                       f"{stats.documents_processed} docs, {stats.tokens_generated} tokens, "
                       f"{stats.processing_time:.2f}s, {stats.errors} errors")
            
            return stats
            
        except Exception as e:
            logger.error(f"Token embedding auto-population failed: {e}")
            stats.processing_time = time.time() - start_time
            stats.errors += 1
            return stats
    
    def _get_documents_missing_token_embeddings(self, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get documents that are missing token embeddings.
        
        Args:
            doc_ids: Optional list of document IDs to check
            
        Returns:
            List of document records missing token embeddings
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            if doc_ids:
                # Check specific documents
                placeholders = ','.join(['?' for _ in doc_ids])
                query = f"""
                    SELECT sd.doc_id, sd.title, sd.abstract, sd.content
                    FROM RAG.SourceDocuments sd
                    LEFT JOIN RAG.DocumentTokenEmbeddings dte ON sd.doc_id = dte.doc_id
                    WHERE sd.doc_id IN ({placeholders}) AND dte.doc_id IS NULL
                """
                cursor.execute(query, doc_ids)
            else:
                # Check all documents
                query = """
                    SELECT sd.doc_id, sd.title, sd.abstract, sd.content
                    FROM RAG.SourceDocuments sd
                    LEFT JOIN RAG.DocumentTokenEmbeddings dte ON sd.doc_id = dte.doc_id
                    WHERE dte.doc_id IS NULL
                """
                cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
        finally:
            cursor.close()
    
    def _process_document_batch(self, documents: List[Dict[str, Any]]) -> TokenEmbeddingStats:
        """
        Process a batch of documents to generate token embeddings.
        
        Args:
            documents: List of document records
            
        Returns:
            TokenEmbeddingStats for this batch
        """
        stats = TokenEmbeddingStats()
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            for doc in documents:
                try:
                    # Extract text content for token embedding generation
                    text_content = self._extract_document_text(doc)
                    
                    if not text_content:
                        logger.warning(f"No text content for document {doc['doc_id']}")
                        continue
                    
                    # Generate token embeddings using ColBERT interface
                    token_embeddings = self.colbert_interface.encode_document(text_content)
                    
                    if not token_embeddings:
                        logger.warning(f"No token embeddings generated for document {doc['doc_id']}")
                        continue
                    
                    # Store token embeddings in database
                    tokens_stored = self._store_token_embeddings(
                        cursor, doc['doc_id'], text_content, token_embeddings
                    )
                    
                    stats.documents_processed += 1
                    stats.tokens_generated += tokens_stored
                    
                    logger.debug(f"Generated {tokens_stored} token embeddings for document {doc['doc_id']}")
                    
                except Exception as e:
                    logger.error(f"Failed to process document {doc['doc_id']}: {e}")
                    stats.errors += 1
            
            # Commit the batch
            connection.commit()
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            connection.rollback()
            stats.errors += len(documents)
        finally:
            cursor.close()
        
        return stats
    
    def _extract_document_text(self, doc: Dict[str, Any]) -> str:
        """
        Extract text content from document record for token embedding generation.
        
        Args:
            doc: Document record
            
        Returns:
            Combined text content
        """
        # Combine available text fields, prioritizing content over abstract over title
        text_parts = []
        
        if doc.get('content'):
            text_parts.append(doc['content'])
        elif doc.get('abstract'):
            text_parts.append(doc['abstract'])
        elif doc.get('title'):
            text_parts.append(doc['title'])
        
        return ' '.join(text_parts).strip()
    
    def _store_token_embeddings(self, cursor, doc_id: str, text_content: str, 
                               token_embeddings: List[List[float]]) -> int:
        """
        Store token embeddings in the database.
        
        Args:
            cursor: Database cursor
            doc_id: Document ID
            text_content: Original text content
            token_embeddings: List of token embeddings
            
        Returns:
            Number of tokens stored
        """
        tokens = text_content.split()[:len(token_embeddings)]  # Match tokens to embeddings
        tokens_stored = 0
        
        for i, embedding in enumerate(token_embeddings):
            try:
                token_text = tokens[i] if i < len(tokens) else f"token_{i}"
                
                # Use insert_vector utility for consistent vector handling
                success = insert_vector(
                    cursor=cursor,
                    table_name="RAG.DocumentTokenEmbeddings",
                    vector_column_name="token_embedding",
                    vector_data=embedding,
                    target_dimension=self.token_dimension,
                    key_columns={
                        "doc_id": doc_id,
                        "token_index": i,
                        "token_text": token_text
                    }
                )
                
                if success:
                    tokens_stored += 1
                else:
                    logger.warning(f"Failed to store token {i} for document {doc_id}")
                    
            except Exception as e:
                logger.error(f"Error storing token {i} for document {doc_id}: {e}")
        
        return tokens_stored
    
    def get_token_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about token embeddings in the database.
        
        Returns:
            Dictionary with token embedding statistics
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Count total documents
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            total_docs = cursor.fetchone()[0]
            
            # Count documents with token embeddings
            cursor.execute("""
                SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings
            """)
            docs_with_tokens = cursor.fetchone()[0]
            
            # Count total tokens
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            total_tokens = cursor.fetchone()[0]
            
            return {
                "total_documents": total_docs,
                "documents_with_token_embeddings": docs_with_tokens,
                "documents_missing_token_embeddings": total_docs - docs_with_tokens,
                "total_token_embeddings": total_tokens,
                "token_dimension": self.token_dimension,
                "coverage_percentage": (docs_with_tokens / total_docs * 100) if total_docs > 0 else 0
            }
            
        finally:
            cursor.close()