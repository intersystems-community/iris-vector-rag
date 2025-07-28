#!/usr/bin/env python3
print("DEBUG: EXECUTING LATEST scripts/validate_colbert_fix.py")
"""
ColBERT Fix Validation Script

This script validates that the ColBERT token embedding fix is working properly
by checking the database state and testing the ColBERT pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connection_manager import get_iris_connection
from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
from common.iris_connection_manager import get_iris_connection
from iris_rag.config.manager import ConfigurationManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_state():
    """Check the current state of the database for ColBERT requirements."""
    logger.info("Checking database state for ColBERT requirements...")
    
    connection = get_iris_connection()
    if not connection:
        logger.error("Failed to connect to database")
        return False
    
    cursor = connection.cursor()
    
    try:
        # Check if DocumentTokenEmbeddings table exists
        cursor.execute("""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings'
        """)
        table_exists = cursor.fetchone()[0] > 0
        
        if not table_exists:
            logger.error("❌ DocumentTokenEmbeddings table does not exist")
            return False
        
        logger.info("✅ DocumentTokenEmbeddings table exists")
        
        # Check if we have any documents
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        logger.info(f"📄 Total documents in SourceDocuments: {doc_count}")
        
        # Check if we have token embeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        logger.info(f"🔤 Total token embeddings: {token_count}")
        
        # Check documents with token embeddings
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentTokenEmbeddings")
        docs_with_tokens = cursor.fetchone()[0]
        logger.info(f"📊 Documents with token embeddings: {docs_with_tokens}")
        
        # Sample a few token embeddings to check format
        cursor.execute("""
            SELECT TOP 3 doc_id, token_index, token_text, 
                   SUBSTRING(CAST(token_embedding AS VARCHAR), 1, 50) as embedding_sample
            FROM RAG.DocumentTokenEmbeddings
            ORDER BY doc_id, token_index
        """)
        samples = cursor.fetchall()
        
        if samples:
            logger.info("📝 Sample token embeddings:")
            for doc_id, token_idx, token_text, embedding_sample in samples:
                logger.info(f"  Doc {doc_id}, Token {token_idx}: '{token_text}' -> {embedding_sample}...")
        
        cursor.close()
        connection.close()
        
        # Determine if ColBERT should work
        if token_count > 0:
            logger.info("✅ Database state looks good for ColBERT")
            return True
        else:
            logger.warning("⚠️  No token embeddings found - ColBERT will not work")
            return False
            
    except Exception as e:
        logger.error(f"Error checking database state: {e}")
        cursor.close()
        connection.close()
        return False

def test_colbert_pipeline():
    """Test the ColBERT pipeline to see if it works."""
    logger.info("Testing ColBERT pipeline...")
    
    try:
        # Initialize ColBERT pipeline
        config_manager = ConfigurationManager()
        connection = get_iris_connection()
        
        pipeline = ColBERTRAGPipeline(
            iris_connector=connection,
            config_manager=config_manager
        )
        
        # Test validation
        logger.info("Running ColBERT validation...")
        validation_result = pipeline.validate_setup()
        
        if validation_result:
            logger.info("✅ ColBERT validation passed!")
            
            # Try a simple query
            logger.info("Testing ColBERT query execution...")
            try:
                result = pipeline.run("What are the effects of BRCA1 mutations?", top_k=3)
                
                if result and result.get("retrieved_documents"):
                    logger.info(f"✅ ColBERT query successful! Retrieved {len(result['retrieved_documents'])} documents")
                    logger.info(f"📝 Answer length: {len(result.get('answer', ''))} characters")
                    return True
                else:
                    logger.warning("⚠️  ColBERT query returned no results")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ ColBERT query failed: {e}")
                return False
        else:
            logger.error("❌ ColBERT validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Error testing ColBERT pipeline: {e}")
        return False

def suggest_fix_actions():
    """Suggest actions to fix ColBERT if it's not working."""
    logger.info("\n" + "="*60)
    logger.info("COLBERT FIX SUGGESTIONS")
    logger.info("="*60)
    
    logger.info("If ColBERT is not working, try these steps:")
    logger.info("")
    logger.info("1. Run the enhanced data loading process:")
    logger.info("   make load-1000")
    logger.info("")
    logger.info("2. Or run the data processing script directly:")
    logger.info("   python scripts/data_processing/process_documents_with_colbert.py")
    logger.info("")
    logger.info("3. Populate missing ColBERT embeddings:")
    logger.info("   python scripts/utilities/populate_missing_colbert_embeddings.py")
    logger.info("")
    logger.info("4. Check database schema:")
    logger.info("   python -m common.db_init_with_indexes")
    logger.info("")

def main():
    """Main validation function."""
    logger.info("🔍 ColBERT Fix Validation Starting...")
    logger.info("="*60)
    
    # Check database state
    db_state_ok = check_database_state()
    
    logger.info("\n" + "-"*40)
    
    # Test ColBERT pipeline
    pipeline_ok = test_colbert_pipeline()
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    if db_state_ok and pipeline_ok:
        logger.info("🎉 SUCCESS: ColBERT fix is working properly!")
        logger.info("✅ Database has token embeddings")
        logger.info("✅ ColBERT pipeline validation passes")
        logger.info("✅ ColBERT queries work correctly")
        logger.info("")
        logger.info("The ultimate_zero_to_ragas_demo.py script should now work with ColBERT!")
        return True
    else:
        logger.error("❌ FAILURE: ColBERT fix needs more work")
        if not db_state_ok:
            logger.error("❌ Database state issues detected")
        if not pipeline_ok:
            logger.error("❌ ColBERT pipeline issues detected")
        
        suggest_fix_actions()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)