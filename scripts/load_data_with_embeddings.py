#!/usr/bin/env python3
"""
Load data with proper embeddings using vector SQL utilities
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.iris_connection_manager import get_iris_connection
from common.utils import get_embedding_func
from common.db_vector_utils import insert_vector
from data.pmc_processor import process_pmc_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_with_embeddings(directory: str, limit: int = 100):
    """Load documents with proper 384-dimensional embeddings."""
    
    # Get embedding function
    embed_func = get_embedding_func()
    logger.info("Using embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dims)")
    
    # Process documents
    logger.info(f"Processing documents from {directory}...")
    documents = list(process_pmc_files(directory, limit=limit))
    logger.info(f"Processed {len(documents)} documents")
    
    # Get connection
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    success_count = 0
    
    try:
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(documents)} documents")
            
            try:
                # Get text for embedding - use abstract or content
                text_to_embed = doc.get("abstract") or doc.get("content") or doc.get("title", "")
                if not text_to_embed:
                    logger.warning(f"No text to embed for doc {doc.get('doc_id')}")
                    continue
                
                # Generate embedding
                embedding = embed_func(text_to_embed)
                
                # Prepare document data
                doc_id = doc.get("doc_id") or doc.get("pmc_id")
                title = doc.get("title", "")[:500]  # Limit title length
                # Use 'content' field for text_content, fallback to abstract
                text_content = doc.get("content") or doc.get("abstract") or ""
                authors = str(doc.get("authors", []))[:500]
                keywords = str(doc.get("keywords", []))[:500]
                
                # Use db_vector_utils.insert_vector() which handles IRIS limitations
                success = insert_vector(
                    cursor=cursor,
                    table_name="RAG.SourceDocuments",
                    vector_column_name="embedding",
                    vector_data=embedding,  # Pass as list of floats
                    target_dimension=384,
                    key_columns={"doc_id": doc_id},
                    additional_data={
                        "title": title,
                        "text_content": text_content,
                        "authors": authors,
                        "keywords": keywords
                    }
                )
                
                if success:
                    success_count += 1
                else:
                    logger.error(f"Failed to insert doc {doc_id}")
                
            except Exception as e:
                logger.error(f"Error loading doc {doc.get('doc_id')}: {e}")
        
        # Commit
        connection.commit()
        logger.info(f"Successfully loaded {success_count}/{len(documents)} documents with embeddings")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
        count = cursor.fetchone()[0]
        logger.info(f"Total documents with embeddings: {count}")
        
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="data/pmc_oas_downloaded")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    
    # Clear existing data first
    logger.info("Clearing existing data...")
    conn = get_iris_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM RAG.DocumentTokenEmbeddings")
    cursor.execute("DELETE FROM RAG.DocumentChunks")
    cursor.execute("DELETE FROM RAG.SourceDocuments")
    conn.commit()
    cursor.close()
    conn.close()
    
    # Load new data
    load_documents_with_embeddings(args.directory, args.limit)