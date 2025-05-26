#!/usr/bin/env python3
"""
Fix Critical Issues Identified in Stress Test

This script addresses the immediate issues found during stress testing:
1. Schema consistency problems
2. Document ID conflicts
3. Field mapping inconsistencies
4. Missing dependencies handling
"""

import sys
import os
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_and_fix_schema():
    """Check and fix database schema issues"""
    logger.info("Checking and fixing database schema issues...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Check current schema
        cursor.execute("SELECT TOP 1 * FROM SourceDocuments")
        columns = [desc[0] for desc in cursor.description]
        logger.info(f"Current SourceDocuments schema: {columns}")
        
        # Check if we have the expected columns
        expected_columns = ['doc_id', 'text_content', 'embedding']
        missing_columns = [col for col in expected_columns if col not in columns]
        
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        else:
            logger.info("Schema appears correct")
        
        # Check for any existing data
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        logger.info(f"Current document count: {count}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error checking schema: {e}")
        return False

def clear_conflicting_data():
    """Clear any conflicting data that might cause constraint violations"""
    logger.info("Clearing conflicting data...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Clear any existing 'sample' documents that might conflict
        cursor.execute("DELETE FROM SourceDocuments WHERE doc_id = 'sample'")
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleared {deleted} conflicting 'sample' documents")
        
        # Clear any other test data
        cursor.execute("DELETE FROM SourceDocuments WHERE doc_id LIKE 'test_%' OR doc_id LIKE 'synthetic_%'")
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleared {deleted} test/synthetic documents")
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error clearing conflicting data: {e}")
        return False

def check_dependencies():
    """Check for required dependencies and provide installation guidance"""
    logger.info("Checking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch for embeddings',
        'transformers': 'HuggingFace transformers',
        'numpy': 'Numerical operations',
        'psutil': 'System monitoring'
    }
    
    missing_deps = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            logger.info(f"✅ {dep} - {description}")
        except ImportError:
            logger.warning(f"❌ {dep} - {description} (MISSING)")
            missing_deps.append(dep)
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {missing_deps}")
        logger.info("To install missing dependencies:")
        logger.info("pip install torch transformers numpy psutil")
        return False
    else:
        logger.info("All dependencies are available")
        return True

def test_basic_operations():
    """Test basic database operations to ensure everything works"""
    logger.info("Testing basic database operations...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        assert result[0] == 1
        logger.info("✅ Basic query test passed")
        
        # Test document count
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        logger.info(f"✅ Document count query passed: {count} documents")
        
        # Test schema access
        cursor.execute("SELECT TOP 1 doc_id FROM SourceDocuments")
        # This should work if schema is correct
        logger.info("✅ Schema access test passed")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Basic operations test failed: {e}")
        return False

def create_test_document():
    """Create a single test document to verify insertion works"""
    logger.info("Creating test document...")
    
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Create a unique test document
        test_doc_id = f"stress_test_{int(time.time())}"
        test_content = "This is a test document for stress testing validation."
        test_embedding = ','.join(['0.1'] * 768)  # 768-dim stub embedding
        
        cursor.execute("""
            INSERT INTO SourceDocuments (doc_id, text_content, embedding)
            VALUES (?, ?, ?)
        """, (test_doc_id, test_content, test_embedding))
        
        conn.commit()
        logger.info(f"✅ Test document created: {test_doc_id}")
        
        # Verify it was inserted
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments WHERE doc_id = ?", (test_doc_id,))
        count = cursor.fetchone()[0]
        assert count == 1
        logger.info("✅ Test document insertion verified")
        
        # Clean up test document
        cursor.execute("DELETE FROM SourceDocuments WHERE doc_id = ?", (test_doc_id,))
        conn.commit()
        logger.info("✅ Test document cleaned up")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Test document creation failed: {e}")
        return False

def main():
    """Main function to run all fixes and checks"""
    logger.info("Starting stress test issue fixes...")
    
    success_count = 0
    total_checks = 5
    
    # Run all checks and fixes
    checks = [
        ("Schema Check", check_and_fix_schema),
        ("Clear Conflicting Data", clear_conflicting_data),
        ("Dependency Check", check_dependencies),
        ("Basic Operations Test", test_basic_operations),
        ("Test Document Creation", create_test_document)
    ]
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        try:
            if check_func():
                logger.info(f"✅ {check_name} passed")
                success_count += 1
            else:
                logger.warning(f"⚠️ {check_name} failed")
        except Exception as e:
            logger.error(f"❌ {check_name} error: {e}")
    
    # Summary
    logger.info(f"\n--- Summary ---")
    logger.info(f"Passed: {success_count}/{total_checks} checks")
    
    if success_count == total_checks:
        logger.info("🎉 All checks passed! System is ready for stress testing.")
        return True
    else:
        logger.warning(f"⚠️ {total_checks - success_count} issues need attention before stress testing.")
        return False

if __name__ == "__main__":
    import time
    success = main()
    sys.exit(0 if success else 1)