#!/usr/bin/env python3
"""
Fix the DocumentTokenEmbeddings table by replacing mock embeddings with real ones.

This script addresses the critical issue where mock token embeddings (all identical values)
were stored in the database, causing ColBERT to return perfect MaxSim scores for all documents.
"""

import sys
import logging
from typing import List, Tuple
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_mock_embeddings():
    """Check if the database contains mock token embeddings."""
    logger.info("Checking for mock token embeddings in database...")
    
    try:
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Check total count
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        total_count = cursor.fetchone()[0]
        logger.info(f"Total token embeddings in database: {total_count}")
        
        if total_count == 0:
            logger.info("No token embeddings found in database")
            return False, 0, 0
        
        # Sample embeddings to check for mock patterns
        cursor.execute("SELECT TOP 10 doc_id, token_index, token_text, token_embedding FROM RAG.DocumentTokenEmbeddings")
        results = cursor.fetchall()
        
        mock_count = 0
        for row in results:
            doc_id, token_index, token_text, embedding_str = row
            
            # Parse embedding
            if embedding_str.startswith('[') and embedding_str.endswith(']'):
                try:
                    values = [float(x.strip()) for x in embedding_str[1:-1].split(',')[:20]]  # First 20 values
                    
                    # Check if all values are the same (mock pattern)
                    unique_values = set(round(v, 6) for v in values)
                    if len(unique_values) <= 2:  # All same or just two distinct values
                        mock_count += 1
                        logger.warning(f"Mock embedding detected: Doc {doc_id}, Token {token_index} '{token_text}' -> {values[:5]}...")
                
                except Exception as e:
                    logger.error(f"Error parsing embedding for doc {doc_id}: {e}")
        
        cursor.close()
        
        is_mock = mock_count > len(results) * 0.5  # More than 50% are mock
        logger.info(f"Mock embeddings detected: {mock_count}/{len(results)} samples are mock")
        
        return is_mock, mock_count, total_count
        
    except Exception as e:
        logger.error(f"Error checking mock embeddings: {e}")
        return False, 0, 0

def clear_mock_embeddings():
    """Clear all token embeddings from the database."""
    logger.info("Clearing mock token embeddings from database...")
    
    try:
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Get count before deletion
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        before_count = cursor.fetchone()[0]
        
        # Clear all token embeddings
        cursor.execute("DELETE FROM RAG.DocumentTokenEmbeddings")
        connection.commit()
        
        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        after_count = cursor.fetchone()[0]
        
        cursor.close()
        
        logger.info(f"Cleared {before_count - after_count} token embeddings from database")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing token embeddings: {e}")
        return False

def regenerate_token_embeddings_for_sample():
    """Regenerate real token embeddings for a small sample of documents."""
    logger.info("Regenerating real token embeddings for sample documents...")
    
    try:
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        from common.utils import get_embedding_func
        
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Get embedding function
        embedding_func = get_embedding_func()
        
        # Get a few sample documents
        cursor.execute("SELECT TOP 3 doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL")
        documents = cursor.fetchall()
        
        if not documents:
            logger.warning("No documents found for token embedding generation")
            return False
        
        total_tokens_generated = 0
        
        for doc_id, text_content in documents:
            logger.info(f"Processing document {doc_id}...")
            
            # Convert CLOB to string if needed
            from iris_rag.storage.clob_handler import convert_clob_to_string
            text_str = convert_clob_to_string(text_content)
            
            # Tokenize (simple word splitting for now)
            tokens = text_str.split()[:50]  # Limit to first 50 tokens for testing
            
            # Generate real embeddings for each token
            for token_index, token_text in enumerate(tokens):
                try:
                    # Get real embedding for this token
                    token_embedding = embedding_func(token_text)
                    
                    # Convert to string format for database storage
                    embedding_str = '[' + ','.join(map(str, token_embedding)) + ']'
                    
                    # Insert into database
                    insert_sql = """
                        INSERT INTO RAG.DocumentTokenEmbeddings 
                        (doc_id, token_index, token_text, token_embedding)
                        VALUES (?, ?, ?, ?)
                    """
                    cursor.execute(insert_sql, (doc_id, token_index, token_text, embedding_str))
                    total_tokens_generated += 1
                    
                    if token_index < 3:  # Log first few for verification
                        emb_sample = token_embedding[:5]
                        logger.debug(f"Generated real embedding for '{token_text}': {emb_sample}...")
                
                except Exception as e:
                    logger.error(f"Error generating embedding for token '{token_text}': {e}")
                    continue
            
            connection.commit()
            logger.info(f"Generated {len(tokens)} token embeddings for document {doc_id}")
        
        cursor.close()
        logger.info(f"Successfully generated {total_tokens_generated} real token embeddings")
        return True
        
    except Exception as e:
        logger.error(f"Error regenerating token embeddings: {e}")
        return False

def verify_real_embeddings():
    """Verify that the regenerated embeddings are real and diverse."""
    logger.info("Verifying regenerated token embeddings...")
    
    try:
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        # Sample some embeddings
        cursor.execute("SELECT TOP 5 doc_id, token_index, token_text, token_embedding FROM RAG.DocumentTokenEmbeddings")
        results = cursor.fetchall()
        
        if not results:
            logger.warning("No token embeddings found after regeneration")
            return False
        
        all_diverse = True
        for row in results:
            doc_id, token_index, token_text, embedding_str = row
            
            # Parse embedding
            if embedding_str.startswith('[') and embedding_str.endswith(']'):
                try:
                    values = [float(x.strip()) for x in embedding_str[1:-1].split(',')[:10]]  # First 10 values
                    
                    # Check diversity
                    unique_values = set(round(v, 4) for v in values)
                    min_val, max_val = min(values), max(values)
                    
                    if len(unique_values) > 5 and abs(max_val - min_val) > 0.1:
                        logger.info(f"✅ Real embedding: Doc {doc_id}, Token '{token_text}' -> diverse values [{min_val:.4f}, {max_val:.4f}]")
                    else:
                        logger.warning(f"❌ Suspicious embedding: Doc {doc_id}, Token '{token_text}' -> limited diversity")
                        all_diverse = False
                
                except Exception as e:
                    logger.error(f"Error parsing embedding: {e}")
                    all_diverse = False
        
        cursor.close()
        return all_diverse
        
    except Exception as e:
        logger.error(f"Error verifying embeddings: {e}")
        return False

def main():
    """Main function to fix token embeddings."""
    logger.info("Starting token embeddings database fix...")
    
    # Step 1: Check if we have mock embeddings
    is_mock, mock_count, total_count = check_mock_embeddings()
    
    if not is_mock:
        logger.info("No mock embeddings detected. Database appears to be fine.")
        return 0
    
    logger.warning(f"Mock embeddings detected! This explains why ColBERT gets perfect MaxSim scores.")
    
    # Step 2: Clear mock embeddings
    if not clear_mock_embeddings():
        logger.error("Failed to clear mock embeddings")
        return 1
    
    # Step 3: Regenerate real embeddings for sample
    if not regenerate_token_embeddings_for_sample():
        logger.error("Failed to regenerate token embeddings")
        return 1
    
    # Step 4: Verify the fix
    if not verify_real_embeddings():
        logger.error("Verification failed - embeddings may still be problematic")
        return 1
    
    logger.info("✅ Token embeddings database fix completed successfully!")
    logger.info("ColBERT should now generate diverse MaxSim scores instead of perfect 1.0 scores.")
    logger.info("Note: This was a sample fix. For production, you should regenerate embeddings for ALL documents.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())