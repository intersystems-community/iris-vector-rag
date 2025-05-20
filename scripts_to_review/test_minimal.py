"""
Minimal test file to verify testcontainer setup.
"""
import pytest
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables directly in the test file
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"

@pytest.mark.force_testcontainer
def test_minimal_testcontainer(iris_testcontainer_connection):
    """Very basic test to verify testcontainer is working."""
    assert iris_testcontainer_connection is not None
    
    # Test a simple query
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == 1
        
        # Create a simple table and insert data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS TestTable (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(100)
            )
        """)
        
        cursor.execute("INSERT INTO TestTable VALUES ('test1', 'Test Name')")
        
        # Verify data was inserted
        cursor.execute("SELECT * FROM TestTable")
        rows = cursor.fetchall()
        assert len(rows) > 0
        logger.info(f"Retrieved row: {rows[0]}")
