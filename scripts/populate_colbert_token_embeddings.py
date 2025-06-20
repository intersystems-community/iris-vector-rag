#!/usr/bin/env python3
"""
Populate DocumentTokenEmbeddings table for ColBERT pipeline.

This script generates token-level embeddings for each document
to enable fine-grained ColBERT retrieval.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import re
from typing import List, Tuple
from common.iris_connection_manager import get_iris_connection
from common.utils import get_embedding_func
from common.db_vector_utils import insert_vector
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenEmbeddingGenerator:
    """Generate token embeddings for ColBERT."""
    
    def __init__(self, max_tokens_per_doc: int = 512):
        self.embedding_func = get_embedding_func()
        self.max_tokens_per_doc = max_tokens_per_doc
        
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words using simple regex."""
        # Simple word tokenization using regex
        # Split on whitespace and punctuation, but keep important terms together
        tokens = re.findall(r'\b\w+\b|[.!?,;:]', text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Skip very short tokens and numbers
            if len(token) > 2 and not token.isdigit():
                filtered_tokens.append(token)
        
        # Limit to max tokens
        return filtered_tokens[:self.max_tokens_per_doc]
    
    def generate_token_embeddings(self, doc_id: str, text: str) -> List[Tuple[int, str, List[float]]]:
        """Generate embeddings for each token in the document."""
        tokens = self.tokenize_text(text)
        
        if not tokens:
            return []
        
        token_embeddings = []
        
        # Generate embeddings for each token
        # In practice, ColBERT would use contextual embeddings, but we'll use
        # individual token embeddings as an approximation
        for i, token in enumerate(tokens):
            try:
                # Generate embedding for the token
                embedding = self.embedding_func(token)
                token_embeddings.append((i, token, embedding))
            except Exception as e:
                logger.warning(f"Failed to generate embedding for token '{token}': {e}")
        
        return token_embeddings

def populate_token_embeddings(limit: int = 100):
    """Populate DocumentTokenEmbeddings table."""
    
    connection = get_iris_connection()
    cursor = connection.cursor()
    generator = TokenEmbeddingGenerator()
    
    try:
        # Get documents that don't have token embeddings yet
        cursor.execute("""
            SELECT d.doc_id, d.title, d.text_content 
            FROM RAG.SourceDocuments d
            WHERE d.doc_id NOT IN (
                SELECT DISTINCT doc_id FROM RAG.DocumentTokenEmbeddings
            )
            AND d.text_content IS NOT NULL
            LIMIT ?
        """, [limit])
        
        documents = cursor.fetchall()
        logger.info(f"Found {len(documents)} documents without token embeddings")
        
        total_tokens = 0
        
        for i, (doc_id, title, content) in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}...")
            
            # Combine title and content
            full_text = f"{title or ''} {content or ''}"
            
            # Generate token embeddings
            token_embeddings = generator.generate_token_embeddings(doc_id, full_text)
            
            # Store token embeddings
            for token_index, token_text, embedding in token_embeddings:
                try:
                    # Use the insert_vector utility to handle IRIS limitations
                    success = insert_vector(
                        cursor=cursor,
                        table_name="RAG.DocumentTokenEmbeddings",
                        vector_column_name="token_embedding",
                        vector_data=embedding,
                        target_dimension=384,  # Using same dimension as document embeddings
                        key_columns={
                            "doc_id": doc_id,
                            "token_index": token_index
                        },
                        additional_data={
                            "token_text": token_text[:500]  # Limit token text length
                        }
                    )
                    
                    if success:
                        total_tokens += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to insert token embedding for '{token_text}': {e}")
            
            # Commit periodically
            if (i + 1) % 10 == 0:
                connection.commit()
                logger.info(f"Committed {total_tokens} token embeddings so far...")
        
        # Final commit
        connection.commit()
        
        logger.info(f"\n✅ Successfully populated {total_tokens} token embeddings")
        
        # Show statistics
        cursor.execute("""
            SELECT COUNT(DISTINCT doc_id) as doc_count, 
                   COUNT(*) as token_count,
                   AVG(LENGTH(token_text)) as avg_token_length
            FROM RAG.DocumentTokenEmbeddings
        """)
        
        row = cursor.fetchone()
        if row:
            logger.info(f"\nToken embedding statistics:")
            logger.info(f"  Documents with token embeddings: {row[0]}")
            logger.info(f"  Total token embeddings: {row[1]}")
            logger.info(f"  Average token length: {row[2]:.1f} characters")
        
        # Show sample tokens
        cursor.execute("""
            SELECT token_text, COUNT(*) as freq
            FROM RAG.DocumentTokenEmbeddings
            WHERE LENGTH(token_text) > 3
            GROUP BY token_text
            ORDER BY freq DESC
            LIMIT 20
        """)
        
        logger.info("\nMost frequent tokens:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]} occurrences")
            
    except Exception as e:
        logger.error(f"Error populating token embeddings: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of documents to process")
    args = parser.parse_args()
    
    logger.info("Populating DocumentTokenEmbeddings table for ColBERT...")
    populate_token_embeddings(args.limit)