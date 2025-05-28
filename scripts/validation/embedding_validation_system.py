#!/usr/bin/env python3
"""
Embedding Validation and Safe Generation System
Implements robust validation and safe embedding generation for IRIS Community Edition
"""

import sys
import json
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add the project root to the path
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingValidator:
    """Validates embedding formats for IRIS Community Edition compatibility"""
    
    @staticmethod
    def validate_embedding_format(embedding: Any) -> Tuple[bool, str]:
        """
        Validate embedding format for IRIS Community Edition VARCHAR storage
        
        Args:
            embedding: The embedding to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if embedding is None
            if embedding is None:
                return False, "Embedding is None"
            
            # Check if embedding is empty string
            if embedding == "":
                return False, "Embedding is empty string"
            
            # If it's a list or numpy array, convert to string format
            if isinstance(embedding, (list, np.ndarray)):
                if len(embedding) == 0:
                    return False, "Embedding array is empty"
                
                # Convert to comma-separated string format
                embedding_str = ",".join(map(str, embedding))
            elif isinstance(embedding, str):
                embedding_str = embedding
            else:
                return False, f"Invalid embedding type: {type(embedding)}"
            
            # Check for problematic characters that cause LIST ERROR
            if '[' in embedding_str or ']' in embedding_str:
                return False, "Embedding contains brackets which cause LIST ERROR"
            
            if '"' in embedding_str:
                return False, "Embedding contains quotes which may cause LIST ERROR"
            
            # Check if it's a valid comma-separated numeric format
            if not re.match(r'^-?\d+\.?\d*(?:,-?\d+\.?\d*)*$', embedding_str):
                return False, "Embedding is not in valid comma-separated numeric format"
            
            # Parse and validate numeric values
            try:
                values = [float(x) for x in embedding_str.split(',')]
            except ValueError as e:
                return False, f"Invalid numeric values in embedding: {e}"
            
            # Check for reasonable embedding dimensions (typically 384, 768, 1024, etc.)
            if len(values) < 100 or len(values) > 2048:
                return False, f"Unusual embedding dimension: {len(values)}"
            
            # Check for reasonable value ranges (embeddings typically in [-1, 1] or similar)
            if any(abs(v) > 100 for v in values):
                return False, "Embedding values outside reasonable range"
            
            # Check for all zeros (likely invalid)
            if all(v == 0 for v in values):
                return False, "Embedding contains all zeros"
            
            return True, "Valid embedding format"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def format_embedding_for_iris(embedding: Any) -> Optional[str]:
        """
        Format embedding for safe storage in IRIS Community Edition VARCHAR column
        
        Args:
            embedding: Raw embedding (list, numpy array, or string)
            
        Returns:
            Formatted embedding string or None if invalid
        """
        try:
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # If already a string, validate and return
            if isinstance(embedding, str):
                is_valid, _ = EmbeddingValidator.validate_embedding_format(embedding)
                return embedding if is_valid else None
            
            # Convert list to comma-separated string
            if isinstance(embedding, list):
                # Ensure all values are numeric
                try:
                    numeric_values = [float(x) for x in embedding]
                except (ValueError, TypeError):
                    return None
                
                # Format as comma-separated string
                embedding_str = ",".join(f"{v:.6f}" for v in numeric_values)
                
                # Validate the formatted string
                is_valid, _ = EmbeddingValidator.validate_embedding_format(embedding_str)
                return embedding_str if is_valid else None
            
            return None
            
        except Exception as e:
            logger.error(f"Error formatting embedding: {e}")
            return None

class SafeEmbeddingGenerator:
    """Generates embeddings with validation and error handling"""
    
    def __init__(self):
        self.embedding_func = None
        self.validator = EmbeddingValidator()
        self.generation_stats = {
            'total_attempts': 0,
            'successful_generations': 0,
            'validation_failures': 0,
            'generation_errors': 0
        }
    
    def initialize_embedding_function(self) -> bool:
        """Initialize the embedding function"""
        try:
            logger.info("🔧 Initializing embedding function...")
            self.embedding_func = get_embedding_model(mock=True)  # Use mock for safety during recovery
            logger.info("✅ Embedding function initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding function: {e}")
            return False
    
    def generate_safe_embedding(self, text: str) -> Optional[str]:
        """
        Generate a safe, validated embedding for the given text
        
        Args:
            text: Text to embed
            
        Returns:
            Validated embedding string or None if generation/validation fails
        """
        self.generation_stats['total_attempts'] += 1
        
        try:
            if not self.embedding_func:
                if not self.initialize_embedding_function():
                    self.generation_stats['generation_errors'] += 1
                    return None
            
            # Generate raw embedding
            raw_embedding = self.embedding_func.encode([text])[0]  # Use encode method and get first result
            
            # Format for IRIS storage
            formatted_embedding = self.validator.format_embedding_for_iris(raw_embedding)
            
            if formatted_embedding is None:
                self.generation_stats['validation_failures'] += 1
                logger.warning(f"Embedding validation failed for text: {text[:50]}...")
                return None
            
            # Final validation
            is_valid, error_msg = self.validator.validate_embedding_format(formatted_embedding)
            
            if not is_valid:
                self.generation_stats['validation_failures'] += 1
                logger.warning(f"Final validation failed: {error_msg}")
                return None
            
            self.generation_stats['successful_generations'] += 1
            return formatted_embedding
            
        except Exception as e:
            self.generation_stats['generation_errors'] += 1
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        stats = self.generation_stats.copy()
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0
        return stats

class DatabaseEmbeddingManager:
    """Manages embedding operations in the database with validation"""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.generator = SafeEmbeddingGenerator()
    
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.conn = get_iris_connection()
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        except Exception as e:
            logger.warning(f"Warning during disconnect: {e}")
    
    def test_database_health(self) -> bool:
        """Test basic database operations to ensure health"""
        try:
            logger.info("🏥 Testing database health...")
            
            # Test basic query
            self.cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            count = self.cursor.fetchone()[0]
            
            # Test sample data retrieval
            self.cursor.execute("""
            SELECT TOP 5 doc_id, title 
            FROM RAG.SourceDocuments 
            WHERE title IS NOT NULL
            """)
            samples = self.cursor.fetchall()
            
            if len(samples) > 0:
                logger.info(f"✅ Database health check passed - {count:,} documents available")
                return True
            else:
                logger.error("❌ Database health check failed - no sample data")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            return False
    
    def regenerate_embeddings_batch(self, batch_size: int = 100, max_batches: int = 5) -> Dict[str, Any]:
        """
        Regenerate embeddings for a small test batch
        
        Args:
            batch_size: Number of documents per batch
            max_batches: Maximum number of batches to process
            
        Returns:
            Results dictionary with statistics and status
        """
        logger.info(f"🔄 Starting embedding regeneration - {batch_size} docs per batch, max {max_batches} batches")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'max_batches': max_batches,
            'batches_processed': 0,
            'documents_processed': 0,
            'embeddings_generated': 0,
            'errors': [],
            'generation_stats': {}
        }
        
        try:
            if not self.generator.initialize_embedding_function():
                results['errors'].append("Failed to initialize embedding function")
                return results
            
            for batch_num in range(max_batches):
                logger.info(f"📦 Processing batch {batch_num + 1}/{max_batches}")
                
                # Get batch of documents without embeddings
                self.cursor.execute(f"""
                SELECT TOP {batch_size} doc_id, title, content 
                FROM RAG.SourceDocuments 
                WHERE (embedding IS NULL OR embedding = '') 
                AND title IS NOT NULL 
                AND content IS NOT NULL
                ORDER BY doc_id
                """)
                
                batch_docs = self.cursor.fetchall()
                
                if not batch_docs:
                    logger.info("No more documents to process")
                    break
                
                batch_success = 0
                batch_errors = 0
                
                for doc_id, title, content in batch_docs:
                    try:
                        # Create text for embedding (title + content sample)
                        embed_text = f"{title}\n{content[:1000]}"  # Limit content length
                        
                        # Generate safe embedding
                        embedding = self.generator.generate_safe_embedding(embed_text)
                        
                        if embedding:
                            # Update database with validated embedding
                            self.cursor.execute("""
                            UPDATE RAG.SourceDocuments 
                            SET embedding = ? 
                            WHERE doc_id = ?
                            """, (embedding, doc_id))
                            
                            batch_success += 1
                            results['embeddings_generated'] += 1
                        else:
                            batch_errors += 1
                            logger.warning(f"Failed to generate embedding for {doc_id}")
                        
                        results['documents_processed'] += 1
                        
                    except Exception as e:
                        batch_errors += 1
                        error_msg = f"Error processing {doc_id}: {e}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
                
                # Commit batch
                self.conn.commit()
                results['batches_processed'] += 1
                
                logger.info(f"✅ Batch {batch_num + 1} completed: {batch_success} success, {batch_errors} errors")
                
                # Safety check - if too many errors, stop
                if batch_errors > batch_success:
                    logger.warning("Too many errors in batch, stopping regeneration")
                    break
            
            # Get final generation statistics
            results['generation_stats'] = self.generator.get_generation_stats()
            
            logger.info(f"🎯 Regeneration completed: {results['embeddings_generated']} embeddings generated")
            
        except Exception as e:
            error_msg = f"Critical error during regeneration: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results

def main():
    """Test the embedding validation and generation system"""
    print("\n" + "="*80)
    print("EMBEDDING VALIDATION AND GENERATION SYSTEM TEST")
    print("="*80)
    
    # Test embedding validation
    validator = EmbeddingValidator()
    
    # Test cases
    test_embeddings = [
        ([0.1, 0.2, 0.3] * 256, "Valid list embedding"),  # 768 dimensions
        ("0.1,0.2,0.3", "Valid string embedding"),
        ("[0.1,0.2,0.3]", "Invalid - contains brackets"),
        ("", "Invalid - empty string"),
        (None, "Invalid - None"),
        ("0.1,0.2,invalid", "Invalid - non-numeric"),
    ]
    
    print("\n🔍 VALIDATION TESTS:")
    for embedding, description in test_embeddings:
        is_valid, message = validator.validate_embedding_format(embedding)
        status = "✅ PASS" if is_valid else "❌ FAIL"
        print(f"  {status} {description}: {message}")
    
    # Test database operations
    manager = DatabaseEmbeddingManager()
    
    if manager.connect():
        print("\n🏥 DATABASE HEALTH CHECK:")
        health_ok = manager.test_database_health()
        
        if health_ok:
            print("\n🔄 SMALL BATCH REGENERATION TEST:")
            results = manager.regenerate_embeddings_batch(batch_size=10, max_batches=1)
            
            print(f"  Processed: {results['documents_processed']} documents")
            print(f"  Generated: {results['embeddings_generated']} embeddings")
            print(f"  Errors: {len(results['errors'])}")
            
            if results['generation_stats']:
                stats = results['generation_stats']
                print(f"  Success Rate: {stats['success_rate']:.2%}")
        
        manager.disconnect()
    else:
        print("❌ Could not connect to database")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()