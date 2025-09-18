#!/usr/bin/env python3
"""
Comprehensive test script for the vector SQL utilities fix.

This script validates that the format_vector_search_sql functions 
generate proper IRIS-compatible SQL without triggering auto-parameterization.
"""

import sys
import os
import re
import logging
from typing import List

# Add the project root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.vector_sql_utils import (
    format_vector_search_sql,
    format_vector_search_sql_with_params,
    validate_vector_string,
    validate_top_k
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_sql_generation():
    """Test that SQL generation produces valid IRIS syntax."""
    
    logger.info("Testing vector SQL generation fix...")
    
    # Test data
    test_vector = "[0.1, -0.2, 0.3, 1.5e-4, -2.7e+3]"
    test_table = "RAG.SourceDocuments"
    test_embedding_dim = 768
    test_top_k = 10
    
    logger.info(f"Test vector: {test_vector}")
    logger.info(f"Test table: {test_table}")
    logger.info(f"Embedding dimension: {test_embedding_dim}")
    logger.info(f"Top K: {test_top_k}")
    
    # Test format_vector_search_sql (string interpolation version)
    logger.info("\n=== Testing format_vector_search_sql ===")
    try:
        sql = format_vector_search_sql(
            table_name=test_table,
            vector_column="embedding",
            vector_string=test_vector,
            embedding_dim=test_embedding_dim,
            top_k=test_top_k,
            id_column="doc_id",
            content_column="text_content"
        )
        
        logger.info("Generated SQL:")
        logger.info(sql)
        
        # Validate that SQL doesn't contain parameter markers
        if ":%qpar" in sql:
            logger.error("❌ FAILED: SQL contains auto-generated parameter markers")
            return False
        
        # Validate that SQL contains expected components
        if f"TOP {test_top_k}" not in sql:
            logger.error("❌ FAILED: SQL missing TOP clause")
            return False
            
        if f"TO_VECTOR('{test_vector}', 'FLOAT', {test_embedding_dim})" not in sql:
            logger.error("❌ FAILED: SQL missing correct TO_VECTOR call")
            return False
            
        logger.info("✅ format_vector_search_sql generates valid SQL")
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception in format_vector_search_sql: {e}")
        return False
    
    # Test format_vector_search_sql_with_params (parameter version)
    logger.info("\n=== Testing format_vector_search_sql_with_params ===")
    try:
        sql_with_params = format_vector_search_sql_with_params(
            table_name=test_table,
            vector_column="embedding",
            embedding_dim=test_embedding_dim,
            top_k=test_top_k,
            id_column="doc_id",
            content_column="text_content"
        )
        
        logger.info("Generated SQL with params:")
        logger.info(sql_with_params)
        
        # Validate that SQL doesn't contain auto-generated parameter markers for embedded values
        if ":%qpar" in sql_with_params:
            logger.error("❌ FAILED: SQL contains auto-generated parameter markers")
            return False
        
        # Validate that SQL contains expected components
        if f"TOP {test_top_k}" not in sql_with_params:
            logger.error("❌ FAILED: SQL missing TOP clause")
            return False
            
        if f"TO_VECTOR(?, 'FLOAT', {test_embedding_dim})" not in sql_with_params:
            logger.error("❌ FAILED: SQL missing correct TO_VECTOR call with parameter")
            return False
            
        # Should contain exactly one ? parameter
        param_count = sql_with_params.count('?')
        if param_count != 1:
            logger.error(f"❌ FAILED: Expected 1 parameter marker, found {param_count}")
            return False
            
        logger.info("✅ format_vector_search_sql_with_params generates valid SQL")
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception in format_vector_search_sql_with_params: {e}")
        return False
    
    return True

def test_validation_functions():
    """Test input validation functions."""
    
    logger.info("\n=== Testing validation functions ===")
    
    # Test validate_vector_string
    valid_vectors = [
        "[0.1, 0.2, 0.3]",
        "[0.1,-0.2,0.3]",
        "[1.5e-4, -2.7e+3]",
        "[0]",
        "[-1.0, 1.0]"
    ]
    
    invalid_vectors = [
        "0.1, 0.2, 0.3",  # Missing brackets
        "[0.1, 0.2, DROP TABLE users; --]",  # SQL injection attempt
        "[0.1, 0.2, abc]",  # Invalid number
        "[]",  # Empty
        "[0.1, 0.2,]",  # Trailing comma
    ]
    
    for vector in valid_vectors:
        if not validate_vector_string(vector):
            logger.error(f"❌ FAILED: Valid vector rejected: {vector}")
            return False
    
    for vector in invalid_vectors:
        if validate_vector_string(vector):
            logger.error(f"❌ FAILED: Invalid vector accepted: {vector}")
            return False
    
    logger.info("✅ validate_vector_string works correctly")
    
    # Test validate_top_k
    valid_top_k = [1, 5, 10, 100, 1000]
    invalid_top_k = [0, -1, 1.5, "10", "DROP TABLE"]
    
    for k in valid_top_k:
        if not validate_top_k(k):
            logger.error(f"❌ FAILED: Valid top_k rejected: {k}")
            return False
    
    for k in invalid_top_k:
        if validate_top_k(k):
            logger.error(f"❌ FAILED: Invalid top_k accepted: {k}")
            return False
    
    logger.info("✅ validate_top_k works correctly")
    
    return True

def test_edge_cases():
    """Test edge cases and potential injection scenarios."""
    
    logger.info("\n=== Testing edge cases ===")
    
    # Test with additional_where clause
    try:
        sql = format_vector_search_sql(
            table_name="RAG.TestTable",
            vector_column="embedding",
            vector_string="[0.1, 0.2]",
            embedding_dim=2,
            top_k=5,
            id_column="doc_id",
            content_column="content",
            additional_where="doc_type = 'test'"
        )
        
        if "doc_type = 'test'" not in sql:
            logger.error("❌ FAILED: additional_where clause not included")
            return False
            
        logger.info("✅ additional_where clause handling works")
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception with additional_where: {e}")
        return False
    
    # Test without content_column
    try:
        sql = format_vector_search_sql(
            table_name="RAG.TestTable",
            vector_column="embedding",
            vector_string="[0.1, 0.2]",
            embedding_dim=2,
            top_k=5,
            id_column="doc_id",
            content_column=None
        )
        
        # Should not contain content column reference
        if ", content" in sql:
            logger.error("❌ FAILED: content column included when None specified")
            return False
            
        logger.info("✅ content_column=None handling works")
        
    except Exception as e:
        logger.error(f"❌ FAILED: Exception with content_column=None: {e}")
        return False
    
    return True

def test_sql_injection_prevention():
    """Test that SQL injection attempts are blocked."""
    
    logger.info("\n=== Testing SQL injection prevention ===")
    
    injection_attempts = [
        ("table_name", "RAG.Test'; DROP TABLE users; --"),
        ("vector_column", "embedding'; DROP TABLE users; --"),
        ("id_column", "doc_id'; DROP TABLE users; --"),
        ("content_column", "content'; DROP TABLE users; --"),
    ]
    
    for param_name, malicious_value in injection_attempts:
        try:
            kwargs = {
                "table_name": "RAG.TestTable",
                "vector_column": "embedding",
                "vector_string": "[0.1, 0.2]",
                "embedding_dim": 2,
                "top_k": 5,
                "id_column": "doc_id",
                "content_column": "content"
            }
            kwargs[param_name] = malicious_value
            
            sql = format_vector_search_sql(**kwargs)
            logger.error(f"❌ FAILED: SQL injection not prevented for {param_name}")
            return False
            
        except ValueError:
            # Expected - injection should be blocked
            continue
        except Exception as e:
            logger.error(f"❌ FAILED: Unexpected exception for {param_name}: {e}")
            return False
    
    logger.info("✅ SQL injection prevention works")
    return True

def main():
    """Run all tests."""
    
    logger.info("🧪 Starting comprehensive vector SQL utilities tests...")
    
    tests = [
        test_validation_functions,
        test_sql_generation,
        test_edge_cases,
        test_sql_injection_prevention
    ]
    
    for test_func in tests:
        if not test_func():
            logger.error("❌ Test suite FAILED")
            return 1
    
    logger.info("✅ All tests PASSED - vector SQL utilities fix is working correctly")
    return 0

if __name__ == "__main__":
    sys.exit(main())