#!/usr/bin/env python3
"""
Health probe script for RAG vector search functionality.

This script validates that the complete RAG and GraphRAG vector search stack
is working correctly with HNSW indexes and safe vector utilities.

Features:
- Connects to IRIS via standard connection utilities
- Ensures HNSW/ACORN-1 indexes exist on vector tables
- Tests safe vector search on both SourceDocuments and Entities tables
- Reports latency and result counts
- Exits with non-zero code on any failures

Usage:
    python3 scripts/check_vector_search.py

Environment:
    Uses standard IRIS connection configuration - no hardcoded values.
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main health probe function."""
    try:
        # Import required modules
        from common.iris_dbapi_connector import get_iris_dbapi_connection
        from common.utils import get_embedding_func
        from common.vector_sql_utils import build_safe_vector_dot_sql, execute_safe_vector_search
        from iris_rag.storage.schema_manager import SchemaManager
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.core.connection import ConnectionManager
        
        logger.info("🔍 Starting RAG vector search health probe...")
        
        # 1. Connect to IRIS database
        logger.info("📡 Connecting to IRIS database...")
        connection = get_iris_dbapi_connection()
        cursor = connection.cursor()
        logger.info("✅ Connected to IRIS successfully")
        
        # 2. Set up managers
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager()
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        # 3. Ensure vector indexes exist
        logger.info("🔧 Ensuring HNSW vector indexes...")
        schema_manager.ensure_all_vector_indexes()
        logger.info("✅ Vector indexes ensured")
        
        # 4. Get embedding function
        logger.info("🤖 Initializing embedding function...")
        embed_func = get_embedding_func()
        logger.info("✅ Embedding function ready")
        
        # 5. Generate test query vector
        test_query = "What are the symptoms of diabetes?"
        logger.info(f"📝 Generating embedding for test query: '{test_query}'")
        query_vector = embed_func(test_query)
        logger.info(f"✅ Generated {len(query_vector)}D embedding vector")
        
        # 6. Test SourceDocuments table
        logger.info("📊 Testing SourceDocuments vector search...")
        source_docs_sql = build_safe_vector_dot_sql(
            table="RAG.SourceDocuments",
            vector_column="embedding",
            id_column="doc_id", 
            extra_columns=["title"],
            top_k=5
        )
        
        start_time = time.time()
        source_results = execute_safe_vector_search(cursor, source_docs_sql, query_vector)
        source_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"✅ SourceDocuments search: {len(source_results)} results in {source_latency:.2f}ms")
        
        # 7. Test Entities table
        logger.info("🏷️  Testing Entities vector search...")
        entities_sql = build_safe_vector_dot_sql(
            table="RAG.Entities",
            vector_column="embedding",
            id_column="entity_id",
            extra_columns=["entity_name", "entity_type"], 
            top_k=3
        )
        
        start_time = time.time()
        entity_results = execute_safe_vector_search(cursor, entities_sql, query_vector)
        entity_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"✅ Entities search: {len(entity_results)} results in {entity_latency:.2f}ms")
        
        # 8. Verify HNSW indexes exist
        logger.info("🔍 Verifying HNSW indexes...")
        index_check_sql = """
            SELECT TABLE_NAME, INDEX_NAME, INDEX_TYPE 
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE INDEX_NAME LIKE '%embedding%' AND INDEX_TYPE = 'HNSW'
        """
        cursor.execute(index_check_sql)
        indexes = cursor.fetchall()
        
        logger.info(f"✅ Found {len(indexes)} HNSW vector indexes:")
        for table_name, index_name, index_type in indexes:
            logger.info(f"   - {table_name}.{index_name} ({index_type})")
        
        # 9. Print summary report
        total_latency = source_latency + entity_latency
        total_results = len(source_results) + len(entity_results)
        
        print("\n" + "="*60)
        print("🎯 RAG VECTOR SEARCH HEALTH PROBE SUMMARY")
        print("="*60)
        print(f"📊 SourceDocuments: {len(source_results)} results ({source_latency:.2f}ms)")
        print(f"🏷️  Entities:        {len(entity_results)} results ({entity_latency:.2f}ms)")
        print(f"🔍 HNSW Indexes:    {len(indexes)} active")
        print(f"⚡ Total Latency:   {total_latency:.2f}ms")
        print(f"📈 Total Results:   {total_results}")
        print("="*60)
        
        # 10. Validate results
        success = True
        if len(indexes) == 0:
            logger.error("❌ No HNSW indexes found - vector search will be slow")
            success = False
            
        if total_results == 0:
            logger.warning("⚠️  No vector search results returned - database may be empty")
            
        if total_latency > 5000:  # 5 seconds
            logger.warning(f"⚠️  High latency detected: {total_latency:.2f}ms")
        
        # 11. Close connection
        cursor.close()
        connection.close()
        
        if success:
            logger.info("🎉 RAG vector search health probe PASSED")
            return 0
        else:
            logger.error("💥 RAG vector search health probe FAILED")
            return 1
            
    except ImportError as e:
        logger.error(f"❌ Import error - missing dependencies: {e}")
        return 1
    except Exception as e:
        logger.error(f"💥 Health probe failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return 1


def check_table_exists(cursor, table_name):
    """Check if a table exists in the database."""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return True
    except Exception:
        return False


def print_sample_results(results, table_name, max_samples=3):
    """Print sample results for debugging."""
    if not results:
        logger.info(f"   No results from {table_name}")
        return
        
    logger.info(f"   Sample results from {table_name}:")
    for i, row in enumerate(results[:max_samples]):
        if len(row) >= 3:
            logger.info(f"     {i+1}. ID: {row[0]}, Score: {row[-1]:.4f}")
        else:
            logger.info(f"     {i+1}. {row}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)