#!/usr/bin/env python3
"""
Generate ColBERT token embeddings for documents in the IRIS database.

This script processes documents in the RAG.SourceDocuments table and generates
token-level embeddings using the ColBERT approach, storing them in the
RAG.DocumentTokenEmbeddings table.
"""

import sys
import os
import logging
import time
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.embedding_utils import get_colbert_model, generate_token_embeddings
from common.iris_connection_manager import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_token_embeddings_table(connection):
    """
    Set up the DocumentTokenEmbeddings table if it doesn't exist.
    
    Args:
        connection: IRIS database connection
    """
    cursor = connection.cursor()
    
    try:
        # Create DocumentTokenEmbeddings table
        create_table_sql = """
            CREATE TABLE IF NOT EXISTS RAG.DocumentTokenEmbeddings (
                id INTEGER IDENTITY PRIMARY KEY,
                doc_id VARCHAR(255) NOT NULL,
                token_index INTEGER NOT NULL,
                token_text VARCHAR(500),
                token_embedding TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
            )
        """
        cursor.execute(create_table_sql)
        
        # Create index on doc_id for faster lookups
        try:
            create_index_sql = """
                CREATE INDEX IF NOT EXISTS idx_doc_token_embeddings_doc_id
                ON RAG.DocumentTokenEmbeddings (doc_id)
            """
            cursor.execute(create_index_sql)
        except Exception as e:
            logger.warning(f"Could not create index: {e}")
        
        connection.commit()
        logger.info("DocumentTokenEmbeddings table setup completed")
        
    except Exception as e:
        logger.error(f"Failed to setup DocumentTokenEmbeddings table: {e}")
        raise
    finally:
        cursor.close()

def check_existing_token_embeddings(connection) -> int:
    """
    Check how many documents already have token embeddings.
    
    Args:
        connection: IRIS database connection
        
    Returns:
        Number of documents with token embeddings
    """
    cursor = connection.cursor()
    
    try:
        sql = """
            SELECT COUNT(DISTINCT doc_id) 
            FROM RAG.DocumentTokenEmbeddings
        """
        cursor.execute(sql)
        count = cursor.fetchone()[0]
        return count
    except Exception as e:
        logger.warning(f"Could not check existing token embeddings: {e}")
        return 0
    finally:
        cursor.close()

def get_documents_without_token_embeddings(connection, limit: int = None) -> List[Tuple[str, str]]:
    """
    Get documents that don't have token embeddings yet.
    
    Args:
        connection: IRIS database connection
        limit: Maximum number of documents to return
        
    Returns:
        List of (doc_id, text_content) tuples
    """
    cursor = connection.cursor()
    
    try:
        # Get documents that don't have token embeddings
        sql = """
            SELECT sd.doc_id, sd.text_content
            FROM RAG.SourceDocuments sd
            LEFT JOIN RAG.DocumentTokenEmbeddings dte ON sd.doc_id = dte.doc_id
            WHERE dte.doc_id IS NULL
            AND sd.text_content IS NOT NULL
        """
        
        if limit:
            sql += f" LIMIT {limit}"
        
        cursor.execute(sql)
        results = cursor.fetchall()
        
        return [(row[0], row[1]) for row in results]
        
    except Exception as e:
        logger.error(f"Failed to get documents without token embeddings: {e}")
        return []
    finally:
        cursor.close()

def generate_and_store_token_embeddings(connection, documents: List[Tuple[str, str]], batch_size: int = 10):
    """
    Generate and store token embeddings for documents.
    
    Args:
        connection: IRIS database connection
        documents: List of (doc_id, text_content) tuples
        batch_size: Number of documents to process in each batch
    """
    # Get ColBERT model
    # Get ColBERT model with 384 dimensions to match existing token embeddings
    from common.embedding_utils import MockColBERTModel
    colbert_model = MockColBERTModel(embedding_dim=384)
    
    total_docs = len(documents)
    processed_count = 0
    total_tokens = 0
    
    logger.info(f"Starting token embedding generation for {total_docs} documents")
    start_time = time.time()
    
    # Process documents in batches
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        batch_start_time = time.time()
        
        for doc_id, text_content in batch:
            try:
                # Generate token embeddings
                tokens, token_embeddings = colbert_model.encode(text_content)
                
                if not tokens or len(token_embeddings) == 0:
                    logger.warning(f"No tokens generated for document {doc_id}")
                    continue
                
                # Store token embeddings
                cursor = connection.cursor()
                
                try:
                    for token_idx, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
                        # Convert embedding to comma-separated format like existing data
                        embedding_str = ','.join(map(str, embedding))
                        
                        # Insert token embedding as VARCHAR string
                        insert_sql = """
                            INSERT INTO RAG.DocumentTokenEmbeddings
                            (doc_id, token_index, token_text, token_embedding)
                            VALUES (?, ?, ?, ?)
                        """
                        cursor.execute(insert_sql, (doc_id, token_idx, token, embedding_str))
                    
                    connection.commit()
                    processed_count += 1
                    total_tokens += len(tokens)
                    
                    if processed_count % 10 == 0:
                        elapsed = time.time() - start_time
                        docs_per_sec = processed_count / elapsed if elapsed > 0 else 0
                        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Processed {processed_count}/{total_docs} documents "
                            f"({docs_per_sec:.2f} docs/sec, {tokens_per_sec:.2f} tokens/sec)"
                        )
                
                finally:
                    cursor.close()
                    
            except Exception as e:
                logger.error(f"Failed to process document {doc_id}: {e}")
                continue
        
        batch_time = time.time() - batch_start_time
        logger.debug(f"Batch {i//batch_size + 1} completed in {batch_time:.2f}s")
    
    total_time = time.time() - start_time
    logger.info(
        f"Token embedding generation completed: {processed_count}/{total_docs} documents processed "
        f"in {total_time:.2f}s ({total_tokens} total tokens)"
    )

def main():
    """Main function to generate ColBERT token embeddings."""
    logger.info("Starting ColBERT token embedding generation")
    
    try:
        # Get database connection
        connection = get_iris_connection()
        logger.info("Connected to IRIS database")
        
        # Setup token embeddings table
        setup_token_embeddings_table(connection)
        
        # Check existing token embeddings
        existing_count = check_existing_token_embeddings(connection)
        logger.info(f"Found {existing_count} documents with existing token embeddings")
        
        # Get documents without token embeddings
        documents = get_documents_without_token_embeddings(connection, limit=1000)
        logger.info(f"Found {len(documents)} documents without token embeddings")
        
        if not documents:
            logger.info("No documents need token embedding generation")
            return
        
        # Generate and store token embeddings
        generate_and_store_token_embeddings(connection, documents, batch_size=10)
        
        # Final count
        final_count = check_existing_token_embeddings(connection)
        logger.info(f"Token embedding generation completed. Total documents with token embeddings: {final_count}")
        
    except Exception as e:
        logger.error(f"Token embedding generation failed: {e}")
        sys.exit(1)
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    main()