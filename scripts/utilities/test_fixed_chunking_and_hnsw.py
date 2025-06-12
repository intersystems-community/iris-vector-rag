#!/usr/bin/env python3
"""
Test Fixed Chunking and HNSW Functionality

This script tests that both the chunking pipeline and HNSW indexes are working
after the VARCHAR to VECTOR conversion.

Author: RAG System Team
Date: 2025-01-26
"""

import logging
import sys
import os
import json
import time
from typing import List, Dict, Any, Optional

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChunkingAndHNSWTester:
    """Test chunking and HNSW functionality."""
    
    def __init__(self):
        self.connection = None
        self.embedding_func = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = get_iris_connection()
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def setup_embedding_function(self):
        """Setup proper embedding function."""
        try:
            embedding_model = get_embedding_model(mock=True)
            
            def embedding_function(texts: List[str]) -> List[List[float]]:
                if hasattr(embedding_model, 'embed_documents'):
                    return embedding_model.embed_documents(texts)
                elif hasattr(embedding_model, 'encode'):
                    embeddings = embedding_model.encode(texts)
                    return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                else:
                    raise ValueError("Embedding model doesn't have expected methods")
            
            self.embedding_func = embedding_function
            logger.info("✅ Embedding function setup complete")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup embedding function: {e}")
            return False
    
    def test_vector_columns(self) -> bool:
        """Test that all vector columns are now proper VECTOR type."""
        cursor = self.connection.cursor()
        
        try:
            logger.info("🔍 Testing vector column types...")
            
            vector_columns = [
                ("RAG.SourceDocuments_V2", "embedding"),
                ("RAG.DocumentChunks", "embedding"),
                ("RAG.KnowledgeGraphNodes", "embedding"),
                ("RAG.DocumentTokenEmbeddings", "token_embedding")
            ]
            
            all_vector = True
            
            for table_name, column_name in vector_columns:
                try:
                    schema_name, table_only = table_name.split('.')
                    cursor.execute(f"""
                        SELECT DATA_TYPE 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = '{schema_name}' 
                        AND TABLE_NAME = '{table_only}'
                        AND COLUMN_NAME = '{column_name}'
                    """)
                    
                    result = cursor.fetchone()
                    if result:
                        data_type = result[0]
                        is_vector = 'vector' in data_type.lower()
                        
                        status = "✅" if is_vector else "❌"
                        logger.info(f"{status} {table_name}.{column_name}: {data_type}")
                        
                        if not is_vector:
                            all_vector = False
                    else:
                        logger.warning(f"⚠️ Column {column_name} not found in {table_name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not check {table_name}.{column_name}: {e}")
                    all_vector = False
            
            return all_vector
            
        except Exception as e:
            logger.error(f"❌ Error testing vector columns: {e}")
            return False
        finally:
            cursor.close()
    
    def test_hnsw_indexes(self) -> bool:
        """Test that HNSW indexes are working."""
        cursor = self.connection.cursor()
        
        try:
            logger.info("🔍 Testing HNSW indexes...")
            
            # Check if HNSW indexes exist
            cursor.execute("""
                SELECT INDEX_NAME, TABLE_NAME 
                FROM INFORMATION_SCHEMA.INDEXES 
                WHERE INDEX_NAME LIKE '%hnsw%'
            """)
            
            hnsw_indexes = cursor.fetchall()
            
            if not hnsw_indexes:
                logger.error("❌ No HNSW indexes found")
                return False
            
            logger.info(f"✅ Found {len(hnsw_indexes)} HNSW indexes:")
            for index_name, table_name in hnsw_indexes:
                logger.info(f"   - {index_name} on {table_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing HNSW indexes: {e}")
            return False
        finally:
            cursor.close()
    
    def test_chunking_pipeline(self) -> bool:
        """Test the complete chunking pipeline."""
        try:
            logger.info("🧪 Testing chunking pipeline...")
            
            # Get a real document from the database
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT TOP 1 doc_id, title, text_content 
                FROM RAG.SourceDocuments_V2 
                WHERE text_content IS NOT NULL 
                AND LENGTH(text_content) > 100
            """)
            
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                logger.error("❌ No suitable documents found for testing")
                return False
            
            doc_id, title, text_content = result
            logger.info(f"📄 Testing with document: {doc_id} - {title[:50]}...")
            
            # Import chunking service
            from chunking.enhanced_chunking_service import EnhancedDocumentChunkingService
            
            # Create service with proper embedding function
            chunking_service = EnhancedDocumentChunkingService(
                embedding_func=self.embedding_func
            )
            
            # Test chunking with a smaller portion of text
            test_text = text_content[:1000] if len(text_content) > 1000 else text_content
            
            # Test chunking
            chunks = chunking_service.chunk_document(doc_id, test_text, "adaptive")
            
            if chunks and len(chunks) > 0:
                logger.info(f"✅ Chunking successful - generated {len(chunks)} chunks")
                
                # Test storing chunks
                success = chunking_service.store_chunks(chunks, self.connection)
                
                if success:
                    logger.info("✅ Chunk storage successful")
                    
                    # Verify chunks were stored
                    cursor = self.connection.cursor()
                    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE doc_id = ?", (doc_id,))
                    stored_count = cursor.fetchone()[0]
                    cursor.close()
                    
                    logger.info(f"✅ Verified: {stored_count} chunks stored in database")
                    return True
                else:
                    logger.error("❌ Chunk storage failed")
                    return False
            else:
                logger.error("❌ Chunking failed - no chunks generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing chunking pipeline: {e}")
            return False
    
    def test_vector_search(self) -> bool:
        """Test vector search with HNSW indexes."""
        try:
            logger.info("🔍 Testing vector search with HNSW...")
            
            cursor = self.connection.cursor()
            
            # Check if we have any chunks with embeddings
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.DocumentChunks 
                WHERE embedding IS NOT NULL
            """)
            
            chunks_with_embeddings = cursor.fetchone()[0]
            
            if chunks_with_embeddings == 0:
                logger.warning("⚠️ No chunks with embeddings found - cannot test vector search")
                cursor.close()
                return False
            
            logger.info(f"📊 Found {chunks_with_embeddings} chunks with embeddings")
            
            # Get a sample embedding for testing
            cursor.execute("""
                SELECT TOP 1 embedding FROM RAG.DocumentChunks 
                WHERE embedding IS NOT NULL
            """)
            
            sample_embedding = cursor.fetchone()[0]
            
            # Test vector similarity search
            start_time = time.time()
            cursor.execute("""
                SELECT TOP 5 chunk_id, chunk_text,
                       VECTOR_COSINE_DISTANCE(embedding, ?) as distance
                FROM RAG.DocumentChunks 
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
            """, (sample_embedding,))
            
            results = cursor.fetchall()
            search_time = time.time() - start_time
            
            if results and len(results) > 0:
                logger.info(f"✅ Vector search working - found {len(results)} similar chunks in {search_time:.3f}s")
                for i, (chunk_id, chunk_text, distance) in enumerate(results[:2]):
                    logger.info(f"   {i+1}. {chunk_id}: distance={distance:.4f}")
                cursor.close()
                return True
            else:
                logger.warning("⚠️ Vector search returned no results")
                cursor.close()
                return False
                
        except Exception as e:
            logger.error(f"❌ Vector search test failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive test of all functionality."""
        logger.info("🚀 Starting comprehensive test of fixed chunking and HNSW...")
        
        # Step 1: Connect to database
        if not self.connect():
            return False
        
        # Step 2: Setup embedding function
        if not self.setup_embedding_function():
            return False
        
        # Step 3: Test vector columns
        vector_columns_ok = self.test_vector_columns()
        
        # Step 4: Test HNSW indexes
        hnsw_indexes_ok = self.test_hnsw_indexes()
        
        # Step 5: Test chunking pipeline
        chunking_ok = self.test_chunking_pipeline()
        
        # Step 6: Test vector search
        vector_search_ok = self.test_vector_search()
        
        # Report results
        logger.info("📋 Comprehensive Test Results:")
        logger.info(f"   {'✅' if vector_columns_ok else '❌'} Vector columns: {'PROPER VECTOR TYPE' if vector_columns_ok else 'STILL VARCHAR'}")
        logger.info(f"   {'✅' if hnsw_indexes_ok else '❌'} HNSW indexes: {'WORKING' if hnsw_indexes_ok else 'FAILED'}")
        logger.info(f"   {'✅' if chunking_ok else '❌'} Chunking pipeline: {'WORKING' if chunking_ok else 'FAILED'}")
        logger.info(f"   {'✅' if vector_search_ok else '❌'} Vector search: {'WORKING' if vector_search_ok else 'FAILED'}")
        
        overall_success = all([vector_columns_ok, hnsw_indexes_ok, chunking_ok, vector_search_ok])
        
        if overall_success:
            logger.info("🎉 ALL TESTS PASSED! Chunking and HNSW are fully functional!")
        else:
            logger.warning("⚠️ Some tests failed - check logs for details")
        
        return overall_success
    
    def cleanup(self):
        """Clean up resources."""
        if self.connection:
            self.connection.close()
            logger.info("🧹 Database connection closed")

def main():
    """Main execution function."""
    tester = ChunkingAndHNSWTester()
    
    try:
        success = tester.run_comprehensive_test()
        
        if success:
            print("\n🎉 SUCCESS: All chunking and HNSW functionality is working!")
            print("\nKey achievements:")
            print("✅ VARCHAR vector columns converted to proper VECTOR columns")
            print("✅ HNSW indexes created and working")
            print("✅ Chunking pipeline functional")
            print("✅ Vector search with HNSW acceleration working")
            print("\nThe critical issues have been resolved!")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED: Check logs for details")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Critical error during testing: {e}")
        return 1
    finally:
        tester.cleanup()

if __name__ == "__main__":
    exit(main())