#!/usr/bin/env python
"""
Direct script to make tests actually run with 1000 documents.

This script takes a step-by-step approach to:
1. Fix common issues that cause test skipping
2. Use real embeddings (no mocking)
3. Show the actual pytest output with tests passing
"""

import os
import sys
import logging
import subprocess
import time

# Configure extensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("make_tests_run.log")
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and log the output"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        shell=True, 
        text=True
    )
    
    # Stream output in real-time
    all_output = []
    for line in iter(process.stdout.readline, ''):
        line = line.rstrip()
        all_output.append(line)
        logger.info(f"► {line}")
    
    process.wait()
    elapsed = time.time() - start_time
    
    result = {
        "command": cmd,
        "return_code": process.returncode,
        "elapsed_time": elapsed,
        "output": "\n".join(all_output)
    }
    
    if process.returncode == 0:
        logger.info(f"✓ Command succeeded (took {elapsed:.2f}s)")
    else:
        logger.error(f"✗ Command failed with return code {process.returncode} (took {elapsed:.2f}s)")
    
    return result

def create_debug_dummy_test():
    """Create a minimal test file for debugging testcontainer connection"""
    test_content = """
import pytest
import logging
import time

logger = logging.getLogger(__name__)

@pytest.mark.force_testcontainer
def test_debug_iris_testcontainer(iris_testcontainer_connection):
    '''Verify IRIS testcontainer is working properly'''
    assert iris_testcontainer_connection is not None
    
    # Try creating a test table
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("CREATE TABLE IF NOT EXISTS DebugTable (id VARCHAR(255) PRIMARY KEY, value VARCHAR(1000))")
        cursor.execute("INSERT INTO DebugTable VALUES ('test1', 'test value')")
        cursor.execute("SELECT * FROM DebugTable")
        rows = cursor.fetchall()
        assert len(rows) > 0
        logger.info(f"Found {len(rows)} rows in DebugTable")
        for row in rows:
            logger.info(f"Row: {row}")
"""
    
    with open("tests/test_debug_iris.py", "w") as f:
        f.write(test_content)
    
    logger.info("Created debug test file: tests/test_debug_iris.py")

def fix_database_schema():
    """Make direct fixes to database schema issues"""
    # Verify db_init.sql has the right column names
    with open("common/db_init.sql", "r") as f:
        content = f.read()
    
    # Check if we need to update SourceDocuments schema
    if "text_content LONGVARCHAR" in content:
        logger.info("Fixing SourceDocuments schema in db_init.sql")
        content = content.replace(
            "text_content LONGVARCHAR", 
            "title VARCHAR(1000),\n    content LONGVARCHAR"
        )
        with open("common/db_init.sql", "w") as f:
            f.write(content)
    else:
        logger.info("Schema in db_init.sql already fixed")
    
    # Check if we need to update KnowledgeGraphNodes schema
    if "content CLOB" in content or "content LONGVARCHAR" in content:
        logger.info("Fixing KnowledgeGraphNodes schema in db_init.sql")
        content = content.replace(
            "content CLOB", 
            "description_text LONGVARCHAR"
        ).replace(
            "content LONGVARCHAR", 
            "description_text LONGVARCHAR"
        )
        with open("common/db_init.sql", "w") as f:
            f.write(content)
    
    # Fix utils.py as well
    with open("tests/utils.py", "r") as f:
        utils_content = f.read()
    
    # Check if we need to update column names in utils.py
    if "content CLOB" in utils_content:
        logger.info("Fixing schema in utils.py")
        utils_content = utils_content.replace(
            "content CLOB", 
            "content LONGVARCHAR"
        )
        with open("tests/utils.py", "w") as f:
            f.write(utils_content)
    
    if "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, content, embedding)" in utils_content:
        logger.info("Fixing insert statement in utils.py")
        utils_content = utils_content.replace(
            "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, content, embedding)",
            "INSERT INTO KnowledgeGraphNodes (node_id, node_type, node_name, description_text, embedding)"
        )
        with open("tests/utils.py", "w") as f:
            f.write(utils_content)
    
    logger.info("Database schema fixes completed")

def main():
    logger.info("=" * 60)
    logger.info("Starting script to make tests run with 1000 documents...")
    logger.info("=" * 60)
    
    # Step 1: Fix database schema issues first
    logger.info("Step 1: Fix database schema issues")
    fix_database_schema()
    
    # Step 2: Create minimal test for debugging
    logger.info("Step 2: Create minimal test file for debugging testcontainer")
    create_debug_dummy_test()
    
    # Step 3: Run a basic test to ensure testcontainer works
    logger.info("Step 3: Running basic test to validate testcontainer setup")
    # Start with a small document count for faster debugging
    os.environ["TEST_IRIS"] = "true"
    os.environ["TEST_DOCUMENT_COUNT"] = "20"
    os.environ["USE_MOCK_EMBEDDINGS"] = "false"  # Use real embeddings as required
    
    result = run_command(
        "poetry run python -m pytest tests/test_debug_iris.py -v",
        "Run debug test with testcontainer"
    )
    
    if result["return_code"] != 0:
        logger.error("Basic testcontainer test failed. Cannot proceed.")
        logger.error("Please fix the testcontainer issues before continuing.")
        return 1
    
    # Step 4: Run GraphRAG test with testcontainer setup
    logger.info("Step 4: Running GraphRAG testcontainer setup test")
    result = run_command(
        "poetry run python -m pytest tests/test_graphrag_with_testcontainer.py::test_iris_testcontainer_setup -v",
        "Run GraphRAG testcontainer setup test"
    )
    
    if result["return_code"] != 0:
        logger.error("GraphRAG testcontainer setup test failed. Cannot proceed.")
        logger.error("Please fix the testcontainer issues in GraphRAG test before continuing.")
        return 1
    
    # Step 5: Run a single data loading test with 100 documents
    logger.info("Step 5: Running GraphRAG data loading test with 100 documents")
    os.environ["TEST_DOCUMENT_COUNT"] = "100"
    
    result = run_command(
        "poetry run python -m pytest tests/test_graphrag_with_testcontainer.py::test_load_pmc_documents -v",
        "Run GraphRAG document loading test"
    )
    
    if result["return_code"] != 0:
        logger.error("Document loading test failed. Cannot proceed to 1000 documents.")
        logger.error("Please fix document loading issues before continuing.")
        return 1
    
    # Step 6: Now run with 1000 documents
    logger.info("Step 6: Running full GraphRAG tests with 1000 documents")
    os.environ["TEST_DOCUMENT_COUNT"] = "1000"
    os.environ["COLLECT_PERFORMANCE_METRICS"] = "true"
    
    result = run_command(
        "poetry run python -m pytest tests/test_graphrag_with_testcontainer.py -v",
        "Run all GraphRAG tests with 1000 documents"
    )
    
    if result["return_code"] == 0:
        logger.info("SUCCESS! All GraphRAG tests passed with 1000 documents!")
    else:
        logger.error("Some GraphRAG tests failed with 1000 documents.")
        logger.error("Check the output above for details.")
    
    # Log a summary of what we've done
    logger.info("=" * 60)
    logger.info("Testing Summary:")
    logger.info("- Fixed database schema issues")
    logger.info("- Created and ran debug test to verify testcontainer setup")
    logger.info("- Ran GraphRAG tests with 1000 documents using real embeddings")
    logger.info("=" * 60)
    
    return result["return_code"]

if __name__ == "__main__":
    sys.exit(main())
