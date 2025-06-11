#!/usr/bin/env python3
"""
Test script to verify GraphRAG entity embedding storage fix.

This script tests the vector embedding storage functionality that was
previously disabled due to IRIS DBAPI driver auto-conversion issues.
"""

import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.embeddings.manager import EmbeddingManager
from common.vector_format_fix import format_vector_for_iris, VectorFormatError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_entity_embedding_storage():
    """Test that entity embeddings can be stored and retrieved correctly."""
    
    try:
        # Initialize managers
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Initialize GraphRAG pipeline
        pipeline = GraphRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager
        )
        
        # Test document
        from iris_rag.core.models import Document
        test_document = Document(
            id="test_embedding_doc_001",
            page_content="This document contains important entities like Python, IRIS database, and machine learning algorithms.",
            metadata={"title": "Test Document for Embedding Storage"}
        )
        
        logger.info("🧪 Testing GraphRAG entity embedding storage...")
        
        # Ingest the document (this should now store embeddings)
        result = pipeline.ingest_documents([test_document])
        
        logger.info(f"✅ Document ingested successfully")
        logger.info(f"   - Entities created: {result.get('entities_created', 0)}")
        logger.info(f"   - Relationships created: {result.get('relationships_created', 0)}")
        logger.info(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
        
        # Verify that entities were stored with embeddings
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check if entities were stored
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.DocumentEntities
                WHERE document_id = ?
            """, [test_document.id])
            entity_count = cursor.fetchone()[0]
            
            # Check if any entities have embeddings
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.DocumentEntities
                WHERE document_id = ? AND embedding IS NOT NULL
            """, [test_document.id])
            entities_with_embeddings = cursor.fetchone()[0]
            
            logger.info(f"📊 Entity storage results:")
            logger.info(f"   - Total entities stored: {entity_count}")
            logger.info(f"   - Entities with embeddings: {entities_with_embeddings}")
            
            if entities_with_embeddings > 0:
                logger.info("✅ SUCCESS: Entity embeddings are being stored correctly!")
                
                # Test entity embedding search
                logger.info("🔍 Testing entity embedding search...")
                
                # Get a sample query embedding
                query_text = "machine learning"
                query_embedding = pipeline.embedding_manager.embed_text(query_text)
                
                # Search for similar entities using the new method
                from iris_rag.storage.iris import IRISStorage
                storage = IRISStorage(connection_manager, config_manager)
                
                similar_entities = storage.search_entity_embeddings(query_embedding, top_k=5)
                
                logger.info(f"🎯 Entity search results for '{query_text}':")
                for i, (entity_id, entity_text, similarity) in enumerate(similar_entities, 1):
                    logger.info(f"   {i}. {entity_text} (ID: {entity_id}, Similarity: {similarity:.4f})")
                
                if similar_entities:
                    logger.info("✅ SUCCESS: Entity embedding search is working!")
                else:
                    logger.warning("⚠️  No similar entities found in search")
                    
            else:
                logger.error("❌ FAILURE: No entities were stored with embeddings")
                return False
                
        finally:
            cursor.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_formatting():
    """Test the vector formatting utilities directly."""
    
    logger.info("🧪 Testing vector formatting utilities...")
    
    try:
        # Test with a sample embedding
        sample_embedding = [0.1, 0.2, 0.3, -0.1, 0.0] * 77  # 385 dimensions (close to typical)
        
        # Test formatting
        formatted = format_vector_for_iris(sample_embedding)
        logger.info(f"✅ Vector formatting successful: {len(formatted)} dimensions")
        
        # Test JSON serialization (what we use for IRIS VECTOR columns)
        import json
        json_str = json.dumps(formatted)
        logger.info(f"✅ JSON serialization successful: {len(json_str)} characters")
        
        # Test deserialization
        deserialized = json.loads(json_str)
        logger.info(f"✅ JSON deserialization successful: {len(deserialized)} dimensions")
        
        return True
        
    except VectorFormatError as e:
        logger.error(f"❌ Vector formatting error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error in vector formatting: {e}")
        return False

def main():
    """Run all tests."""
    
    logger.info("🚀 Starting GraphRAG embedding storage tests...")
    
    # Test 1: Vector formatting utilities
    if not test_vector_formatting():
        logger.error("❌ Vector formatting tests failed")
        return False
    
    # Test 2: Entity embedding storage
    if not test_entity_embedding_storage():
        logger.error("❌ Entity embedding storage tests failed")
        return False
    
    logger.info("🎉 All tests passed! GraphRAG entity embedding storage is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)