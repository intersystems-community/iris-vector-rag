#!/usr/bin/env python
"""
Verify Real Data Testing for RAG Templates

This script verifies that the IRIS database contains at least 1000 real PMC documents,
checks that the documents have proper embeddings, runs a simple vector search query
to verify functionality, and reports detailed diagnostics about the database state.

Usage:
    python scripts/verify_real_data_testing.py [--min-docs 1000] [--verbose]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
try:
    from common.iris_connector import get_iris_connection, IRISConnectionError
    from common.embedding_utils import get_embedding_model
    from common.utils import get_embedding_func
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

# Configure logging
def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging with appropriate level based on verbose flag."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create logger
    logger = logging.getLogger("verify_real_data")
    logger.setLevel(log_level)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def verify_database_connection(logger: logging.Logger) -> Tuple[bool, Any]:
    """
    Verify connection to the IRIS database.
    
    Returns:
        Tuple of (success, connection)
    """
    logger.info("Verifying connection to IRIS database...")
    
    try:
        # Try to get a real connection (not a mock)
        connection = get_iris_connection(use_mock=False)
        
        if connection is None:
            logger.error("Failed to connect to IRIS database.")
            return False, None
        
        # Verify this is a real connection, not a mock
        is_mock = hasattr(connection, '_cursor') and hasattr(connection._cursor, 'stored_docs')
        if is_mock:
            logger.error("Connected to a mock database, not a real IRIS instance.")
            connection.close()
            return False, None
        
        # Test the connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result and result[0] == 1:
                logger.info("✅ Successfully connected to IRIS database.")
                return True, connection
            else:
                logger.error("Database connection test failed.")
                connection.close()
                return False, None
    
    except IRISConnectionError as e:
        logger.error(f"IRIS connection error: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Error verifying database connection: {e}")
        return False, None

def verify_document_count(logger: logging.Logger, connection, min_docs: int = 1000) -> bool:
    """
    Verify that the database contains at least the minimum number of documents.
    
    Args:
        logger: Logger instance
        connection: IRIS connection
        min_docs: Minimum number of documents required
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info(f"Verifying database has at least {min_docs} real PMC documents...")
    
    try:
        with connection.cursor() as cursor:
            # Try with RAG schema qualification first
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2")
                count = cursor.fetchone()[0]
                schema = "RAG.SourceDocuments_V2"
            except Exception:
                try:
                    # Try without schema qualification
                    cursor.execute("SELECT COUNT(*) FROM SourceDocuments_V2")
                    count = cursor.fetchone()[0]
                    schema = "SourceDocuments_V2"
                except Exception as e:
                    logger.error(f"Error querying document count: {e}")
                    return False
            
            logger.info(f"Found {count} documents in {schema}.")
            
            if count < min_docs:
                logger.error(f"Insufficient documents: found {count}, need at least {min_docs}.")
                return False
            
            # Check if these are real PMC documents by looking for PMC IDs
            try:
                # Sample some documents to check if they have PMC IDs
                cursor.execute(f"SELECT TOP 10 doc_id FROM {schema}")
                sample_ids = [row[0] for row in cursor.fetchall()]
                
                pmc_count = sum(1 for doc_id in sample_ids if "PMC" in doc_id)
                if pmc_count == 0:
                    logger.warning("No PMC document IDs found in sample. These may be synthetic documents.")
                    
                    # Check content for PMC references
                    cursor.execute(f"SELECT TOP 10 content FROM {schema}")
                    sample_contents = [row[0] for row in cursor.fetchall()]
                    
                    pmc_content_count = sum(1 for content in sample_contents if content and "PMC" in content)
                    if pmc_content_count == 0:
                        logger.error("No PMC references found in document content. These appear to be synthetic documents.")
                        return False
                    else:
                        logger.info(f"Found {pmc_content_count}/10 documents with PMC references in content.")
                else:
                    logger.info(f"Found {pmc_count}/10 documents with PMC IDs.")
            
            except Exception as e:
                logger.error(f"Error checking for PMC documents: {e}")
                return False
            
            logger.info(f"✅ Database verification passed: {count} documents available.")
            return True
    
    except Exception as e:
        logger.error(f"Error verifying document count: {e}")
        return False

def verify_embeddings(logger: logging.Logger, connection, min_docs: int = 1000) -> bool:
    """
    Verify that documents have proper embeddings.
    
    Args:
        logger: Logger instance
        connection: IRIS connection
        min_docs: Minimum number of documents required
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying document embeddings...")
    
    try:
        with connection.cursor() as cursor:
            # Try with RAG schema qualification first
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE embedding IS NOT NULL")
                count = cursor.fetchone()[0]
                schema = "RAG.SourceDocuments_V2"
            except Exception:
                try:
                    # Try without schema qualification
                    cursor.execute("SELECT COUNT(*) FROM SourceDocuments_V2 WHERE embedding IS NOT NULL")
                    count = cursor.fetchone()[0]
                    schema = "SourceDocuments_V2"
                except Exception as e:
                    logger.error(f"Error querying embedding count: {e}")
                    return False
            
            logger.info(f"Found {count} documents with embeddings in {schema}.")
            
            if count < min_docs:
                logger.error(f"Insufficient documents with embeddings: found {count}, need at least {min_docs}.")
                return False
            
            # Check embedding format and dimensions
            try:
                cursor.execute(f"SELECT TOP 5 embedding FROM {schema} WHERE embedding IS NOT NULL")
                sample_embeddings = [row[0] for row in cursor.fetchall()]
                
                dimensions = []
                for embedding_str in sample_embeddings:
                    try:
                        # Try to parse the embedding
                        if embedding_str.startswith('[') and embedding_str.endswith(']'):
                            # JSON format
                            embedding = json.loads(embedding_str)
                        else:
                            # Python list literal format
                            embedding = eval(embedding_str)
                        
                        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                            dimensions.append(len(embedding))
                        else:
                            logger.error(f"Invalid embedding format: {embedding_str[:100]}...")
                            return False
                    except Exception as e:
                        logger.error(f"Error parsing embedding: {e}")
                        return False
                
                if not dimensions:
                    logger.error("No valid embeddings found.")
                    return False
                
                # Check if all embeddings have the same dimension
                if len(set(dimensions)) > 1:
                    logger.warning(f"Inconsistent embedding dimensions: {dimensions}")
                else:
                    logger.info(f"Embeddings have consistent dimension: {dimensions[0]}")
                
                logger.info(f"✅ Embedding verification passed: {count} documents have valid embeddings.")
                return True
            
            except Exception as e:
                logger.error(f"Error checking embedding format: {e}")
                return False
    
    except Exception as e:
        logger.error(f"Error verifying embeddings: {e}")
        return False

def verify_vector_search(logger: logging.Logger, connection) -> bool:
    """
    Verify that vector search functionality works.
    
    Args:
        logger: Logger instance
        connection: IRIS connection
        
    Returns:
        True if verification passed, False otherwise
    """
    logger.info("Verifying vector search functionality...")
    
    try:
        # Get a real embedding function
        embedding_func = get_embedding_func(mock=False)
        if embedding_func is None:
            logger.error("Failed to get embedding function.")
            return False
        
        # Generate a test query embedding
        test_query = "What are the symptoms of diabetes?"
        logger.info(f"Test query: '{test_query}'")
        
        try:
            query_embedding = embedding_func(test_query)
            if not query_embedding or not isinstance(query_embedding, list):
                logger.error(f"Invalid query embedding: {query_embedding}")
                return False
            
            logger.info(f"Generated query embedding with dimension {len(query_embedding)}")
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return False
        
        # Try to perform a vector search
        try:
            with connection.cursor() as cursor:
                # Try with RAG schema qualification first
                try:
                    schema = "RAG.SourceDocuments_V2"
                    cursor.execute(f"SELECT TOP 1 * FROM {schema} WHERE embedding IS NOT NULL")
                except Exception:
                    schema = "SourceDocuments_V2"
                
                # Convert embedding to string
                query_embedding_str = json.dumps(query_embedding)
                
                # Try different vector search approaches
                search_methods = [
                    {
                        "name": "Cosine similarity with JSON_ARRAY_REAL",
                        "sql": f"""
                            SELECT TOP 5 doc_id, 
                                   VectorSimilarityCosine(
                                       JSON_ARRAY_REAL(embedding), 
                                       JSON_ARRAY_REAL(?)) AS similarity
                            FROM {schema}
                            WHERE embedding IS NOT NULL
                            ORDER BY similarity DESC
                        """
                    },
                    {
                        "name": "Direct cosine similarity",
                        "sql": f"""
                            SELECT TOP 5 doc_id, 
                                   VectorSimilarityCosine(embedding, ?) AS similarity
                            FROM {schema}
                            WHERE embedding IS NOT NULL
                            ORDER BY similarity DESC
                        """
                    },
                    {
                        "name": "Stored procedure vector search",
                        "sql": f"""
                            CALL VectorSearch(?, 5)
                        """
                    }
                ]
                
                success = False
                for method in search_methods:
                    try:
                        logger.info(f"Trying vector search method: {method['name']}")
                        cursor.execute(method['sql'], (query_embedding_str,))
                        results = cursor.fetchall()
                        
                        if results and len(results) > 0:
                            logger.info(f"✅ Vector search successful using {method['name']}")
                            logger.info(f"Top result: {results[0]}")
                            success = True
                            break
                    except Exception as e:
                        logger.warning(f"Method failed: {e}")
                
                if not success:
                    logger.error("All vector search methods failed.")
                    return False
                
                return True
        
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error verifying vector search: {e}")
        return False

def generate_diagnostics_report(logger: logging.Logger, connection, output_dir: str = "test_results") -> str:
    """
    Generate a detailed diagnostics report about the database state.
    
    Args:
        logger: Logger instance
        connection: IRIS connection
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    logger.info("Generating diagnostics report...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"real_data_verification_{timestamp}.json")
        
        report = {
            "timestamp": timestamp,
            "database": {},
            "tables": {},
            "sample_documents": [],
            "vector_search_test": {}
        }
        
        # Get database information
        with connection.cursor() as cursor:
            try:
                cursor.execute("SELECT $SYSTEM.Version.GetVersion()")
                version = cursor.fetchone()[0]
                report["database"]["version"] = version
            except Exception as e:
                logger.warning(f"Error getting database version: {e}")
                report["database"]["version"] = "Unknown"
            
            try:
                cursor.execute("SELECT $NAMESPACE")
                namespace = cursor.fetchone()[0]
                report["database"]["namespace"] = namespace
            except Exception as e:
                logger.warning(f"Error getting namespace: {e}")
                report["database"]["namespace"] = "Unknown"
        
        # Get table information
        tables_to_check = ["SourceDocuments_V2", "DocumentTokenEmbeddings"]
        schemas_to_check = ["", "RAG."]
        
        for schema in schemas_to_check:
            for table in tables_to_check:
                full_table_name = f"{schema}{table}"
                try:
                    with connection.cursor() as cursor:
                        # Check if table exists
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {full_table_name}")
                            count = cursor.fetchone()[0]
                            
                            # Get column information
                            cursor.execute(f"SELECT TOP 1 * FROM {full_table_name}")
                            columns = [column[0] for column in cursor.description]
                            
                            report["tables"][full_table_name] = {
                                "exists": True,
                                "row_count": count,
                                "columns": columns
                            }
                            
                            # For SourceDocuments, get embedding stats
                            if table == "SourceDocuments_V2":
                                try:
                                    cursor.execute(f"SELECT COUNT(*) FROM {full_table_name} WHERE embedding IS NOT NULL")
                                    with_embedding = cursor.fetchone()[0]
                                    report["tables"][full_table_name]["with_embedding"] = with_embedding
                                    report["tables"][full_table_name]["without_embedding"] = count - with_embedding
                                except Exception:
                                    pass
                        except Exception:
                            report["tables"][full_table_name] = {
                                "exists": False
                            }
                except Exception as e:
                    logger.warning(f"Error checking table {full_table_name}: {e}")
        
        # Get sample documents
        try:
            with connection.cursor() as cursor:
                # Find the SourceDocuments table
                source_docs_table = None
                for table_name, table_info in report["tables"].items():
                    if table_info.get("exists", False) and table_name.endswith("SourceDocuments_V2"):
                        source_docs_table = table_name
                        break
                
                if source_docs_table:
                    cursor.execute(f"SELECT TOP 5 doc_id, content FROM {source_docs_table}")
                    for row in cursor.fetchall():
                        doc_id, content = row
                        # Truncate content for the report
                        truncated_content = content[:500] + "..." if content and len(content) > 500 else content
                        report["sample_documents"].append({
                            "doc_id": doc_id,
                            "content_preview": truncated_content
                        })
        except Exception as e:
            logger.warning(f"Error getting sample documents: {e}")
        
        # Run a simple vector search test
        try:
            embedding_func = get_embedding_func(mock=False)
            if embedding_func:
                test_query = "What are the symptoms of diabetes?"
                query_embedding = embedding_func(test_query)
                
                if query_embedding:
                    report["vector_search_test"]["query"] = test_query
                    report["vector_search_test"]["embedding_dimension"] = len(query_embedding)
                    
                    # Try to perform a vector search
                    with connection.cursor() as cursor:
                        # Find the SourceDocuments table
                        source_docs_table = None
                        for table_name, table_info in report["tables"].items():
                            if table_info.get("exists", False) and table_name.endswith("SourceDocuments_V2"):
                                source_docs_table = table_name
                                break
                        
                        if source_docs_table:
                            try:
                                query_embedding_str = json.dumps(query_embedding)
                                cursor.execute(f"""
                                    SELECT TOP 3 doc_id, 
                                           VectorSimilarityCosine(
                                               JSON_ARRAY_REAL(embedding), 
                                               JSON_ARRAY_REAL(?)) AS similarity
                                    FROM {source_docs_table}
                                    WHERE embedding IS NOT NULL
                                    ORDER BY similarity DESC
                                """, (query_embedding_str,))
                                
                                results = []
                                for row in cursor.fetchall():
                                    doc_id, similarity = row
                                    results.append({
                                        "doc_id": doc_id,
                                        "similarity": similarity
                                    })
                                
                                report["vector_search_test"]["results"] = results
                                report["vector_search_test"]["success"] = len(results) > 0
                            except Exception as e:
                                report["vector_search_test"]["error"] = str(e)
                                report["vector_search_test"]["success"] = False
        except Exception as e:
            logger.warning(f"Error running vector search test: {e}")
            report["vector_search_test"]["error"] = str(e)
            report["vector_search_test"]["success"] = False
        
        # Write report to file
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Diagnostics report saved to: {report_path}")
        return report_path
    
    except Exception as e:
        logger.error(f"Error generating diagnostics report: {e}")
        return ""

def main():
    """Main function to verify real data testing."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Verify real data testing for RAG templates.")
    parser.add_argument("--min-docs", type=int, default=1000, help="Minimum number of documents required")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Directory for test reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    logger.info("Starting real data verification...")
    
    # Track overall success
    success = True
    connection = None
    
    try:
        # Step 1: Verify database connection
        logger.info("Step 1: Verifying database connection...")
        connection_success, connection = verify_database_connection(logger)
        if not connection_success:
            logger.error("Database connection verification failed.")
            return 1
        
        # Step 2: Verify document count
        logger.info(f"Step 2: Verifying document count (min: {args.min_docs})...")
        if not verify_document_count(logger, connection, args.min_docs):
            logger.error("Document count verification failed.")
            success = False
        
        # Step 3: Verify embeddings
        logger.info("Step 3: Verifying document embeddings...")
        if not verify_embeddings(logger, connection, args.min_docs):
            logger.error("Embedding verification failed.")
            success = False
        
        # Step 4: Verify vector search
        logger.info("Step 4: Verifying vector search functionality...")
        if not verify_vector_search(logger, connection):
            logger.error("Vector search verification failed.")
            success = False
        
        # Step 5: Generate diagnostics report
        logger.info("Step 5: Generating diagnostics report...")
        report_path = generate_diagnostics_report(logger, connection, args.output_dir)
        if not report_path:
            logger.error("Failed to generate diagnostics report.")
            success = False
        
        # Final status
        if success:
            logger.info("✅ All verification steps passed successfully.")
            return 0
        else:
            logger.error("❌ Some verification steps failed. Please check the logs and diagnostics report for details.")
            return 1
    
    finally:
        # Close connection
        if connection:
            connection.close()

if __name__ == "__main__":
    sys.exit(main())