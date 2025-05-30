#!/usr/bin/env python3
"""
Populate ColBERT Token Embeddings Script

This script populates the DocumentTokenEmbeddings table with token-level embeddings
for existing documents in the RAG.SourceDocuments_V2 table.
"""

import os
import sys
import time
import logging
import json
from typing import List, Dict, Any

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from colbert.doc_encoder import get_colbert_doc_encoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_documents_without_token_embeddings(iris_connector, limit: int = 100) -> List[Dict[str, Any]]:
    """Get documents that don't have token embeddings yet."""
    try:
        cursor = iris_connector.cursor()
        
        # First get doc_ids that don't have token embeddings
        query_doc_ids = f"""
        SELECT TOP {limit} s.doc_id
        FROM RAG.SourceDocuments_V2 s
        LEFT JOIN RAG.DocumentTokenEmbeddings t ON s.doc_id = t.doc_id
        WHERE t.doc_id IS NULL
        """
        
        cursor.execute(query_doc_ids)
        doc_ids = [row[0] for row in cursor.fetchall()]
        
        documents = []
        
        # For each doc_id, fetch the text content separately
        for doc_id in doc_ids:
            try:
                cursor.execute("SELECT text_content FROM RAG.SourceDocuments_V2 WHERE doc_id = ?", (doc_id,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    documents.append({
                        "doc_id": doc_id,
                        "text_content": result[0]
                    })
            except Exception as e:
                logger.warning(f"Error fetching text for doc {doc_id}: {e}")
                continue
        
        cursor.close()
        logger.info(f"Found {len(documents)} documents without token embeddings")
        return documents
        
    except Exception as e:
        logger.error(f"Error getting documents without token embeddings: {e}")
        return []

def insert_token_embeddings(iris_connector, doc_id: str, tokens: List[str], token_embeddings: List[List[float]]) -> bool:
    """Insert token embeddings for a document."""
    try:
        cursor = iris_connector.cursor()
        
        # Insert each token embedding
        for i, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
            # Convert embedding to comma-separated string
            embedding_str = ','.join(map(str, embedding))
            
            insert_sql = """
            INSERT INTO RAG.DocumentTokenEmbeddings 
            (doc_id, token_sequence_index, token_text, token_embedding, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """
            
            metadata = json.dumps({"token_index": i, "token_length": len(token)})
            
            cursor.execute(insert_sql, (doc_id, i, token, embedding_str, metadata))
        
        cursor.close()
        return True
        
    except Exception as e:
        logger.error(f"Error inserting token embeddings for {doc_id}: {e}")
        return False

def populate_token_embeddings(iris_connector, batch_size: int = 10, max_documents: int = 100):
    """Populate token embeddings for documents."""
    logger.info(f"Starting token embeddings population (max {max_documents} documents, batch size {batch_size})")
    
    # Create ColBERT document encoder (using mock for now)
    doc_encoder = get_colbert_doc_encoder(mock=True, embedding_dim=128)
    
    total_processed = 0
    total_tokens_created = 0
    
    while total_processed < max_documents:
        # Get batch of documents without token embeddings
        remaining = max_documents - total_processed
        current_batch_size = min(batch_size, remaining)
        
        documents = get_documents_without_token_embeddings(iris_connector, current_batch_size)
        
        if not documents:
            logger.info("No more documents without token embeddings found")
            break
        
        logger.info(f"Processing batch of {len(documents)} documents...")
        
        for doc in documents:
            try:
                doc_id = doc["doc_id"]
                text_content = doc["text_content"]
                
                # Limit text length for performance
                text_content = text_content[:2000] if text_content else ""
                
                if not text_content.strip():
                    logger.warning(f"Skipping document {doc_id} - no text content")
                    continue
                
                # Generate token embeddings
                tokens, token_embeddings = doc_encoder.encode(text_content)
                
                if not tokens or not token_embeddings:
                    logger.warning(f"No token embeddings generated for document {doc_id}")
                    continue
                
                # Insert token embeddings
                success = insert_token_embeddings(iris_connector, doc_id, tokens, token_embeddings)
                
                if success:
                    total_tokens_created += len(tokens)
                    logger.info(f"Created {len(tokens)} token embeddings for document {doc_id}")
                else:
                    logger.error(f"Failed to insert token embeddings for document {doc_id}")
                
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc.get('doc_id', 'unknown')}: {e}")
                total_processed += 1
                continue
        
        # Small delay between batches
        time.sleep(0.1)
    
    logger.info(f"Token embeddings population completed:")
    logger.info(f"  - Documents processed: {total_processed}")
    logger.info(f"  - Total tokens created: {total_tokens_created}")
    
    return total_processed, total_tokens_created

def verify_token_embeddings(iris_connector) -> Dict[str, Any]:
    """Verify the token embeddings were created successfully."""
    try:
        cursor = iris_connector.cursor()
        
        # Count total token embeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        total_tokens = cursor.fetchone()[0]
        
        # Count documents with token embeddings
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings")
        docs_with_tokens = cursor.fetchone()[0]
        
        # Count null embeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE token_embedding IS NULL")
        null_embeddings = cursor.fetchone()[0]
        
        # Get sample token embedding to verify format
        cursor.execute("SELECT TOP 1 token_embedding FROM RAG.DocumentTokenEmbeddings WHERE token_embedding IS NOT NULL")
        sample_result = cursor.fetchone()
        sample_embedding = sample_result[0] if sample_result else None
        
        cursor.close()
        
        verification_result = {
            "total_tokens": total_tokens,
            "documents_with_tokens": docs_with_tokens,
            "null_embeddings": null_embeddings,
            "valid_embeddings": total_tokens - null_embeddings,
            "sample_embedding_length": len(sample_embedding.split(',')) if sample_embedding else 0
        }
        
        logger.info(f"Token embeddings verification: {verification_result}")
        return verification_result
        
    except Exception as e:
        logger.error(f"Error verifying token embeddings: {e}")
        return {"error": str(e)}

def main():
    """Main function."""
    logger.info("ColBERT Token Embeddings Population Starting...")
    
    try:
        # Get database connection
        iris_connector = get_iris_connection()
        if not iris_connector:
            raise ConnectionError("Failed to get IRIS connection")
        
        # Check current state
        logger.info("Checking current token embeddings state...")
        initial_state = verify_token_embeddings(iris_connector)
        
        if initial_state.get("valid_embeddings", 0) > 0:
            logger.info(f"Found {initial_state['valid_embeddings']} existing token embeddings")
            user_input = input("Do you want to add more token embeddings? (y/n): ")
            if user_input.lower() != 'y':
                logger.info("Skipping token embeddings population")
                iris_connector.close()
                return
        
        # Populate token embeddings
        processed, tokens_created = populate_token_embeddings(
            iris_connector, 
            batch_size=10, 
            max_documents=100
        )
        
        # Verify results
        logger.info("Verifying token embeddings...")
        final_state = verify_token_embeddings(iris_connector)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TOKEN EMBEDDINGS POPULATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Documents processed: {processed}")
        logger.info(f"Tokens created: {tokens_created}")
        logger.info(f"Total token embeddings: {final_state.get('total_tokens', 0)}")
        logger.info(f"Documents with tokens: {final_state.get('documents_with_tokens', 0)}")
        logger.info(f"Valid embeddings: {final_state.get('valid_embeddings', 0)}")
        
        if final_state.get("valid_embeddings", 0) > 0:
            logger.info("✅ Token embeddings population successful!")
            logger.info("You can now test the optimized ColBERT pipeline with real data")
        else:
            logger.warning("⚠️  No valid token embeddings created")
        
        iris_connector.close()
        
    except Exception as e:
        logger.error(f"❌ Fatal error during token embeddings population: {e}", exc_info=True)

if __name__ == "__main__":
    main()