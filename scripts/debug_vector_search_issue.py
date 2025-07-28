#!/usr/bin/env python3
"""
Debug Vector Search Issue

This script systematically diagnoses the vector search problem where documents
are inserted successfully but vector search returns 0 results.

Investigation areas:
1. Driver type detection and capabilities
2. Vector search SQL generation
3. Database transaction isolation
4. Vector indexing and data visibility
"""

import sys
import os
import logging
import json
import time
from typing import List, Dict, Any
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.vector_store_iris import IRISVectorStore
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.core.models import Document
from common.connection_singleton import get_shared_iris_connection, reset_shared_connection
from common.db_driver_utils import get_driver_type, get_driver_capabilities
from common.vector_sql_utils import execute_driver_aware_vector_search

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_test_documents(count: int = 5) -> List[Document]:
    """Create simple test documents for debugging."""
    documents = []
    for i in range(count):
        doc = Document(
            id=f"debug_doc_{i+1:03d}",
            page_content=f"This is test document {i+1} about medical research and COVID-19 treatment options.",
            metadata={
                "source": f"debug_source_{i+1}",
                "category": "medical",
                "debug_test": True
            }
        )
        documents.append(doc)
    return documents

def debug_driver_detection():
    """Debug driver type detection and capabilities."""
    logger.info("=== DRIVER DETECTION DEBUG ===")
    
    # Get connection and detect driver
    connection = get_shared_iris_connection()
    driver_type = get_driver_type(connection)
    capabilities = get_driver_capabilities(driver_type)
    
    logger.info(f"Connection type: {type(connection)}")
    logger.info(f"Detected driver type: {driver_type}")
    logger.info(f"Driver capabilities: {capabilities}")
    
    # Check autocommit status
    try:
        autocommit_status = getattr(connection, 'autocommit', 'Not available')
        logger.info(f"Autocommit status: {autocommit_status}")
    except Exception as e:
        logger.warning(f"Could not check autocommit status: {e}")
    
    return driver_type, capabilities

def debug_schema_and_data():
    """Debug schema setup and data insertion."""
    logger.info("=== SCHEMA AND DATA DEBUG ===")
    
    # Initialize components
    connection = get_shared_iris_connection()
    config_manager = ConfigurationManager()
    
    # Create connection manager wrapper
    class ConnectionManager:
        def __init__(self, connection):
            self._connection = connection
        def get_connection(self):
            return self._connection
    
    connection_manager = ConnectionManager(connection)
    schema_manager = SchemaManager(connection_manager, config_manager)
    
    # Ensure schema exists
    schema_manager.ensure_table_schema('SourceDocuments')
    
    # Clear existing test data
    cursor = connection.cursor()
    try:
        cursor.execute("DELETE FROM RAG.SourceDocuments WHERE metadata LIKE '%debug_test%'")
        connection.commit()
        logger.info("Cleared existing debug test data")
    except Exception as e:
        logger.warning(f"Could not clear test data: {e}")
    finally:
        cursor.close()
    
    # Create vector store
    vector_store = IRISVectorStore(
        config_manager=config_manager,
        schema_manager=schema_manager,
        connection_manager=connection_manager
    )
    
    # Create and insert test documents
    documents = create_test_documents(5)
    logger.info(f"Created {len(documents)} test documents")
    
    # Generate simple embeddings for testing
    test_embeddings = []
    for i, doc in enumerate(documents):
        # Create simple but valid embeddings (384 dimensions)
        embedding = [0.1 + (i * 0.01)] * 384
        test_embeddings.append(embedding)
    
    # Insert documents with embeddings
    try:
        added_ids = vector_store.add_documents(documents, test_embeddings)
        logger.info(f"Successfully added documents: {added_ids}")
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        return None, None
    
    # Verify documents were inserted
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE metadata LIKE '%debug_test%'")
        doc_count = cursor.fetchone()[0]
        logger.info(f"Documents in database: {doc_count}")
        
        # Check for embeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE metadata LIKE '%debug_test%' AND embedding IS NOT NULL")
        embedding_count = cursor.fetchone()[0]
        logger.info(f"Documents with embeddings: {embedding_count}")
        
        # Sample a document to check data
        cursor.execute("SELECT ID, TEXT_CONTENT, embedding FROM RAG.SourceDocuments WHERE metadata LIKE '%debug_test%' LIMIT 1")
        sample_row = cursor.fetchone()
        if sample_row:
            doc_id, content, embedding = sample_row
            logger.info(f"Sample document - ID: {doc_id}, Content length: {len(content) if content else 0}")
            logger.info(f"Sample embedding type: {type(embedding)}, length: {len(str(embedding)) if embedding else 0}")
        
    except Exception as e:
        logger.error(f"Error checking inserted data: {e}")
    finally:
        cursor.close()
    
    return vector_store, test_embeddings[0]  # Return first embedding for search testing

def debug_vector_search_sql():
    """Debug vector search SQL generation."""
    logger.info("=== VECTOR SEARCH SQL DEBUG ===")
    
    connection = get_shared_iris_connection()
    driver_type = get_driver_type(connection)
    
    # Test SQL generation
    from common.vector_sql_utils import get_driver_aware_vector_search_sql
    
    test_embedding = [0.1] * 384
    embedding_str = '[' + ','.join(map(str, test_embedding)) + ']'
    
    try:
        sql, uses_parameters = get_driver_aware_vector_search_sql(
            table_name="RAG.SourceDocuments",
            vector_column="embedding",
            embedding_dim=384,
            top_k=5,
            id_column="ID",
            content_column="TEXT_CONTENT",
            driver_type=driver_type
        )
        
        logger.info(f"Generated SQL (uses_parameters={uses_parameters}):")
        logger.info(sql)
        
        # Test SQL execution
        cursor = connection.cursor()
        try:
            if uses_parameters:
                logger.info("Executing with parameters...")
                cursor.execute(sql, [embedding_str])
            else:
                logger.info("Executing with string interpolation...")
                interpolated_sql = sql.format(vector_string=embedding_str)
                logger.info(f"Interpolated SQL: {interpolated_sql}")
                cursor.execute(interpolated_sql)
            
            results = cursor.fetchall()
            logger.info(f"Raw SQL execution results: {len(results)} rows")
            for i, row in enumerate(results[:3]):  # Show first 3 results
                logger.info(f"  Row {i+1}: {row}")
                
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
        finally:
            cursor.close()
            
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")

def debug_vector_store_search(vector_store, query_embedding):
    """Debug vector store search method."""
    logger.info("=== VECTOR STORE SEARCH DEBUG ===")
    
    if not vector_store or not query_embedding:
        logger.error("Vector store or query embedding not available")
        return
    
    try:
        # Test similarity search
        logger.info("Testing similarity_search_by_embedding...")
        results = vector_store.similarity_search_by_embedding(
            query_embedding=query_embedding,
            top_k=5
        )
        
        logger.info(f"Vector store search results: {len(results)} documents")
        for i, (doc, score) in enumerate(results):
            logger.info(f"  Result {i+1}: ID={doc.id}, Score={score:.4f}, Content={doc.page_content[:50]}...")
            
    except Exception as e:
        logger.error(f"Vector store search failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def debug_transaction_isolation():
    """Debug transaction isolation issues."""
    logger.info("=== TRANSACTION ISOLATION DEBUG ===")
    
    connection = get_shared_iris_connection()
    
    # Check transaction isolation level
    cursor = connection.cursor()
    try:
        # Try to get transaction isolation level (IRIS-specific)
        cursor.execute("SELECT $SYSTEM.SQL.GetTransactionIsolationLevel()")
        isolation_level = cursor.fetchone()[0]
        logger.info(f"Transaction isolation level: {isolation_level}")
    except Exception as e:
        logger.warning(f"Could not get isolation level: {e}")
    
    # Check for uncommitted transactions
    try:
        cursor.execute("SELECT $SYSTEM.SQL.GetTransactionLevel()")
        transaction_level = cursor.fetchone()[0]
        logger.info(f"Transaction level: {transaction_level}")
    except Exception as e:
        logger.warning(f"Could not get transaction level: {e}")
    
    # Force commit to ensure data visibility
    try:
        connection.commit()
        logger.info("Forced commit executed")
    except Exception as e:
        logger.warning(f"Could not force commit: {e}")
    
    cursor.close()

def main():
    """Main debugging function."""
    logger.info("Starting vector search debugging...")
    
    # Reset connection to ensure clean state
    reset_shared_connection()
    
    try:
        # Step 1: Debug driver detection
        driver_type, capabilities = debug_driver_detection()
        
        # Step 2: Debug schema and data insertion
        vector_store, query_embedding = debug_schema_and_data()
        
        # Step 3: Debug transaction isolation
        debug_transaction_isolation()
        
        # Step 4: Debug SQL generation and execution
        debug_vector_search_sql()
        
        # Step 5: Debug vector store search
        debug_vector_store_search(vector_store, query_embedding)
        
        logger.info("=== DEBUGGING COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Debugging failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()