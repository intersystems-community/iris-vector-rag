#!/usr/bin/env python3
"""
Interim Script: Regenerate ColBERT Token Embeddings

This script urgently regenerates real token-level embeddings in the RAG.DocumentTokenEmbeddings table
to unblock the ColBERT pipeline. It provides an immediate solution while a long-term reconciliation-based
architecture is being developed.

Usage:
    # Process a sample of 10 documents for testing
    python regenerate_colbert_token_embeddings_interim.py --sample-size 10
    
    # Process all documents (with confirmation prompt)
    python regenerate_colbert_token_embeddings_interim.py
    
    # Process all documents without confirmation (for automation)
    python regenerate_colbert_token_embeddings_interim.py --no-confirm

WARNING: This script uses simple space-splitting tokenization as an interim solution.
A proper ColBERT tokenizer should be used in the final architectural implementation.
"""

import sys
import os
import logging
import argparse
import time
from typing import List, Optional, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.connection import ConnectionManager
    from common.utils import get_embedding_func, get_config_value
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ColBERTTokenEmbeddingGenerator:
    """
    Interim generator for ColBERT token embeddings using simple tokenization.
    """
    
    def __init__(self):
        """Initialize the generator with configuration and database connections."""
        self.config_manager = ConfigurationManager()
        self.connection_manager = ConnectionManager(self.config_manager)
        self.embedding_func = None
        self.embedding_dimension = None
        self._setup_embedding_function()
        
    def _setup_embedding_function(self):
        """Setup the embedding function and determine dimension."""
        try:
            self.embedding_func = get_embedding_func()
            # Get dimension from config, fallback to 384 if not specified
            self.embedding_dimension = get_config_value("embedding_model.dimension", 384)
            logger.info(f"✅ Embedding function initialized with dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding function: {e}")
            raise
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Simple space-splitting tokenization (interim solution).
        
        WARNING: This is a temporary tokenization strategy. A proper ColBERT tokenizer
        should be used in the final architectural implementation.
        
        Args:
            text: Text content to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []
        
        # Simple space-based tokenization
        tokens = text.strip().split()
        
        # Filter out empty tokens and limit length for performance
        tokens = [token for token in tokens if token.strip()]
        
        # Limit to reasonable number of tokens for interim solution
        max_tokens = get_config_value("colbert.max_tokens_per_document", 512)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            logger.debug(f"Truncated tokens to {max_tokens} for performance")
        
        return tokens
    
    def _generate_token_embeddings(self, tokens: List[str]) -> List[List[float]]:
        """
        Generate embeddings for each token using the configured embedding function.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token embeddings
        """
        if not tokens:
            return []
        
        try:
            # Generate embeddings for all tokens at once for efficiency
            embeddings = self.embedding_func(tokens)
            
            # Ensure we have the right format
            if not isinstance(embeddings, list):
                logger.error(f"Expected list of embeddings, got {type(embeddings)}")
                return []
            
            # Validate embedding dimensions
            for i, embedding in enumerate(embeddings):
                if not isinstance(embedding, list) or len(embedding) != self.embedding_dimension:
                    logger.warning(f"Invalid embedding at index {i}: expected {self.embedding_dimension}D, got {len(embedding) if isinstance(embedding, list) else type(embedding)}")
                    # Create zero vector as fallback
                    embeddings[i] = [0.0] * self.embedding_dimension
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to generate token embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dimension for _ in tokens]
    
    def _format_embedding_for_iris(self, embedding: List[float]) -> str:
        """
        Format embedding as string for IRIS VECTOR storage.
        
        Args:
            embedding: List of float values
            
        Returns:
            Formatted string for IRIS
        """
        # Convert to comma-separated string format expected by IRIS
        return ','.join([f'{x:.10f}' for x in embedding])
    
    def _clear_existing_token_embeddings(self, doc_id: str, cursor) -> bool:
        """
        Clear existing token embeddings for a document to ensure idempotency.
        
        Args:
            doc_id: Document ID
            cursor: Database cursor
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor.execute(
                "DELETE FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?",
                [doc_id]
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear existing token embeddings for {doc_id}: {e}")
            return False
    
    def _store_token_embeddings(self, doc_id: str, tokens: List[str], embeddings: List[List[float]], cursor) -> bool:
        """
        Store token embeddings in the database.
        
        Args:
            doc_id: Document ID
            tokens: List of tokens
            embeddings: List of token embeddings
            cursor: Database cursor
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for token_index, (token, embedding) in enumerate(zip(tokens, embeddings)):
                embedding_str = self._format_embedding_for_iris(embedding)
                
                cursor.execute("""
                    INSERT INTO RAG.DocumentTokenEmbeddings 
                    (doc_id, token_index, token_text, token_embedding)
                    VALUES (?, ?, ?, TO_VECTOR(?))
                """, [doc_id, token_index, token, embedding_str])
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store token embeddings for {doc_id}: {e}")
            return False
    
    def _process_document(self, doc_id: str, text_content: str, cursor) -> bool:
        """
        Process a single document to generate and store token embeddings.
        
        Args:
            doc_id: Document ID
            text_content: Document text content
            cursor: Database cursor
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing token embeddings for idempotency
            if not self._clear_existing_token_embeddings(doc_id, cursor):
                return False
            
            # Tokenize the text content
            tokens = self._tokenize_text(text_content)
            if not tokens:
                logger.warning(f"⚠️ No tokens found for document {doc_id}")
                return True  # Not an error, just empty content
            
            # Generate token embeddings
            embeddings = self._generate_token_embeddings(tokens)
            if len(embeddings) != len(tokens):
                logger.error(f"❌ Embedding count mismatch for {doc_id}: {len(embeddings)} embeddings for {len(tokens)} tokens")
                return False
            
            # Store token embeddings
            if not self._store_token_embeddings(doc_id, tokens, embeddings, cursor):
                return False
            
            logger.debug(f"✅ Processed document {doc_id}: {len(tokens)} tokens embedded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing document {doc_id}: {e}")
            return False
    
    def _get_documents_to_process(self, sample_size: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Get documents from SourceDocuments table to process.
        
        Args:
            sample_size: If specified, limit to this many documents
            
        Returns:
            List of (doc_id, text_content) tuples
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            if sample_size:
                # Use TOP for IRIS SQL (not LIMIT)
                # Note: Cannot filter TEXT fields in WHERE clause due to IRIS limitations
                cursor.execute("""
                    SELECT TOP ? doc_id, text_content
                    FROM RAG.SourceDocuments
                    WHERE text_content IS NOT NULL
                    ORDER BY doc_id
                """, [sample_size])
            else:
                cursor.execute("""
                    SELECT doc_id, text_content
                    FROM RAG.SourceDocuments
                    WHERE text_content IS NOT NULL
                    ORDER BY doc_id
                """)
            
            documents = cursor.fetchall()
            logger.info(f"📊 Found {len(documents)} documents to process")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch documents: {e}")
            return []
        finally:
            cursor.close()
    
    def _verify_generated_embeddings(self, sample_count: int = 5) -> bool:
        """
        Verify that token embeddings were generated correctly by sampling a few.
        
        Args:
            sample_count: Number of embeddings to sample for verification
            
        Returns:
            True if verification passes, False otherwise
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Get a sample of generated token embeddings
            cursor.execute("""
                SELECT TOP ? doc_id, token_text, token_embedding 
                FROM RAG.DocumentTokenEmbeddings 
                ORDER BY doc_id, token_index
            """, [sample_count])
            
            samples = cursor.fetchall()
            
            if not samples:
                logger.error("❌ No token embeddings found for verification")
                return False
            
            logger.info(f"🔍 Verification: Sampled {len(samples)} token embeddings:")
            
            for doc_id, token_text, token_embedding in samples:
                # Parse the vector string to check dimensions
                try:
                    if token_embedding:
                        # Convert IRIS vector format back to list for verification
                        embedding_str = str(token_embedding)
                        if embedding_str.startswith('[') and embedding_str.endswith(']'):
                            embedding_values = embedding_str[1:-1].split(',')
                            embedding_dim = len(embedding_values)
                            first_few_values = embedding_values[:3]
                            logger.info(f"  📄 {doc_id} | '{token_text}' | {embedding_dim}D | [{', '.join(first_few_values)}...]")
                        else:
                            logger.info(f"  📄 {doc_id} | '{token_text}' | Vector format: {embedding_str[:50]}...")
                    else:
                        logger.warning(f"  ⚠️ {doc_id} | '{token_text}' | NULL embedding")
                except Exception as e:
                    logger.warning(f"  ⚠️ {doc_id} | '{token_text}' | Error parsing embedding: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
        finally:
            cursor.close()
    
    def process_documents(self, sample_size: Optional[int] = None, batch_size: int = 100) -> bool:
        """
        Process documents to generate token embeddings.
        
        Args:
            sample_size: If specified, process only this many documents
            batch_size: Number of documents to process per batch
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Starting ColBERT token embedding generation...")
        logger.warning("⚠️ Using simple space-splitting tokenization (interim solution)")
        logger.warning("⚠️ A proper ColBERT tokenizer should be used in the final implementation")
        
        # Get documents to process
        documents = self._get_documents_to_process(sample_size)
        if not documents:
            logger.error("❌ No documents found to process")
            return False
        
        total_docs = len(documents)
        processed_count = 0
        error_count = 0
        
        connection = self.connection_manager.get_connection()
        
        try:
            # Process documents in batches
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                batch_start = i + 1
                batch_end = min(i + batch_size, total_docs)
                
                logger.info(f"📦 Processing batch {batch_start}-{batch_end}/{total_docs}...")
                
                cursor = connection.cursor()
                
                try:
                    for doc_id, text_content in batch:
                        if self._process_document(doc_id, text_content, cursor):
                            processed_count += 1
                        else:
                            error_count += 1
                        
                        # Log progress every 10 documents
                        if processed_count % 10 == 0:
                            logger.info(f"📈 Progress: {processed_count}/{total_docs} documents processed")
                    
                    # Commit batch
                    connection.commit()
                    logger.info(f"✅ Batch {batch_start}-{batch_end} committed successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_start}-{batch_end}: {e}")
                    connection.rollback()
                    error_count += len(batch)
                finally:
                    cursor.close()
        
        except Exception as e:
            logger.error(f"❌ Fatal error during processing: {e}")
            return False
        
        # Final statistics
        logger.info(f"📊 Processing completed:")
        logger.info(f"  ✅ Successfully processed: {processed_count} documents")
        logger.info(f"  ❌ Errors: {error_count} documents")
        logger.info(f"  📈 Success rate: {(processed_count/total_docs)*100:.1f}%")
        
        # Verify generated embeddings
        if processed_count > 0:
            logger.info("🔍 Verifying generated embeddings...")
            if self._verify_generated_embeddings():
                logger.info("✅ Verification passed - embeddings look diverse and valid")
            else:
                logger.warning("⚠️ Verification had issues - please check the results")
        
        return processed_count > 0


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Regenerate ColBERT token embeddings (interim solution)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a sample of 10 documents for testing
  python regenerate_colbert_token_embeddings_interim.py --sample-size 10
  
  # Process all documents (with confirmation prompt)
  python regenerate_colbert_token_embeddings_interim.py
  
  # Process all documents without confirmation (for automation)
  python regenerate_colbert_token_embeddings_interim.py --no-confirm

WARNING: This script uses simple space-splitting tokenization as an interim solution.
A proper ColBERT tokenizer should be used in the final architectural implementation.
        """
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Process only the first N documents (for testing)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process per batch (default: 100)"
    )
    
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompt when processing all documents"
    )
    
    args = parser.parse_args()
    
    try:
        generator = ColBERTTokenEmbeddingGenerator()
        
        # Confirmation prompt for processing all documents
        if not args.sample_size and not args.no_confirm:
            response = input("\n⚠️ You are about to process ALL documents in the database. This may take a long time.\nContinue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("❌ Operation cancelled by user")
                return 1
        
        # Process documents
        start_time = time.time()
        success = generator.process_documents(
            sample_size=args.sample_size,
            batch_size=args.batch_size
        )
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"⏱️ Total processing time: {duration:.2f} seconds")
        
        if success:
            logger.info("🎉 ColBERT token embedding generation completed successfully!")
            return 0
        else:
            logger.error("❌ ColBERT token embedding generation failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("❌ Operation cancelled by user (Ctrl+C)")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())