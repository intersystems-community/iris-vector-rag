"""
Test for SQLCODE: <-104> error with negative values in vector insertions.

This test specifically targets the issue where insert_vector() fails when
vectors contain negative values, causing SQLCODE: <-104> during reconciliation.
"""

import pytest
import logging
from common.db_vector_utils import insert_vector
from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

# Test constants
TEST_TABLE_NAME = "RAG.TestVectorTable"
TEST_VECTOR_COLUMN = "test_embedding"
TEST_DIMENSION = 5


@pytest.fixture(scope="function")
def real_iris_connection():
    """Get a real IRIS connection for testing vector operations."""
    conn = get_iris_connection()
    if conn is None:
        pytest.skip("Real IRIS connection not available")
    
    # Test the connection
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        if result is None or result[0] != 1:
            conn.close()
            pytest.skip("IRIS connection test failed")
    except Exception as e:
        conn.close()
        pytest.skip(f"IRIS connection test failed: {e}")
    
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def test_vector_table_setup(real_iris_connection):
    """Create a test table for vector insertion testing."""
    cursor = real_iris_connection.cursor()
    
    # Drop table if exists
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}")
        real_iris_connection.commit()
    except Exception:
        pass
    
    # Create test table
    cursor.execute(f"""
        CREATE TABLE {TEST_TABLE_NAME} (
            test_id VARCHAR(255) PRIMARY KEY,
            test_name VARCHAR(255),
            {TEST_VECTOR_COLUMN} VECTOR(FLOAT, {TEST_DIMENSION})
        )
    """)
    real_iris_connection.commit()
    
    yield cursor
    
    # Cleanup
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {TEST_TABLE_NAME}")
        real_iris_connection.commit()
    except Exception:
        pass
    cursor.close()


class TestVectorNegativeValues:
    """Test cases for vector insertion with negative values."""
    
    def test_insert_vector_with_positive_values_should_succeed(self, real_iris_connection, test_vector_table_setup):
        """RED: Test that positive values work (baseline test)."""
        cursor = test_vector_table_setup
        
        positive_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result = insert_vector(
            cursor=cursor,
            table_name=TEST_TABLE_NAME,
            vector_column_name=TEST_VECTOR_COLUMN,
            vector_data=positive_vector,
            target_dimension=TEST_DIMENSION,
            key_columns={"test_id": "positive_test"},
            additional_data={"test_name": "positive_values"}
        )
        
        assert result is True, "Positive vector insertion should succeed"
        real_iris_connection.commit()
        
        # Verify insertion
        cursor.execute(f"SELECT test_id FROM {TEST_TABLE_NAME} WHERE test_id = ?", ("positive_test",))
        row = cursor.fetchone()
        assert row is not None, "Positive vector should be inserted successfully"
    
    def test_insert_vector_with_negative_values_should_succeed(self, real_iris_connection, test_vector_table_setup):
        """RED: Test that negative values work (this should initially fail)."""
        cursor = test_vector_table_setup
        
        # Vector with negative values that typically cause SQLCODE: <-104>
        negative_vector = [-0.066, 0.123, -0.045, 0.234, -0.012]
        
        result = insert_vector(
            cursor=cursor,
            table_name=TEST_TABLE_NAME,
            vector_column_name=TEST_VECTOR_COLUMN,
            vector_data=negative_vector,
            target_dimension=TEST_DIMENSION,
            key_columns={"test_id": "negative_test"},
            additional_data={"test_name": "negative_values"}
        )
        
        assert result is True, "Negative vector insertion should succeed"
        real_iris_connection.commit()
        
        # Verify insertion
        cursor.execute(f"SELECT test_id FROM {TEST_TABLE_NAME} WHERE test_id = ?", ("negative_test",))
        row = cursor.fetchone()
        assert row is not None, "Negative vector should be inserted successfully"
    
    def test_insert_vector_with_mixed_values_should_succeed(self, real_iris_connection, test_vector_table_setup):
        """RED: Test that mixed positive/negative values work."""
        cursor = test_vector_table_setup
        
        # Vector with mix of positive, negative, and zero values
        mixed_vector = [-0.5, 0.0, 0.3, -0.1, 0.8]
        
        result = insert_vector(
            cursor=cursor,
            table_name=TEST_TABLE_NAME,
            vector_column_name=TEST_VECTOR_COLUMN,
            vector_data=mixed_vector,
            target_dimension=TEST_DIMENSION,
            key_columns={"test_id": "mixed_test"},
            additional_data={"test_name": "mixed_values"}
        )
        
        assert result is True, "Mixed vector insertion should succeed"
        real_iris_connection.commit()
        
        # Verify insertion
        cursor.execute(f"SELECT test_id FROM {TEST_TABLE_NAME} WHERE test_id = ?", ("mixed_test",))
        row = cursor.fetchone()
        assert row is not None, "Mixed vector should be inserted successfully"
    
    def test_insert_vector_with_very_small_negative_values_should_succeed(self, real_iris_connection, test_vector_table_setup):
        """RED: Test that very small negative values work (edge case)."""
        cursor = test_vector_table_setup
        
        # Vector with very small negative values that might cause parsing issues
        small_negative_vector = [-1e-6, -1e-8, 1e-7, -1e-5, 1e-4]
        
        result = insert_vector(
            cursor=cursor,
            table_name=TEST_TABLE_NAME,
            vector_column_name=TEST_VECTOR_COLUMN,
            vector_data=small_negative_vector,
            target_dimension=TEST_DIMENSION,
            key_columns={"test_id": "small_negative_test"},
            additional_data={"test_name": "small_negative_values"}
        )
        
        assert result is True, "Small negative vector insertion should succeed"
        real_iris_connection.commit()
        
        # Verify insertion
        cursor.execute(f"SELECT test_id FROM {TEST_TABLE_NAME} WHERE test_id = ?", ("small_negative_test",))
        row = cursor.fetchone()
        assert row is not None, "Small negative vector should be inserted successfully"


def test_reproduce_reconciliation_negative_vector_issue(real_iris_connection, test_vector_table_setup):
    """
    RED: Reproduce the specific issue from reconciliation where negative vectors
    cause SQLCODE: <-104> errors.
    
    This test simulates the exact scenario that fails during reconciliation.
    """
    cursor = test_vector_table_setup
    
    # This is the type of vector that causes issues during reconciliation
    # Generated by np.random.rand() and then potentially modified to have negative values
    problematic_vector = [-0.066, 0.123, -0.045, 0.234, -0.012]
    
    logger.info(f"Testing problematic vector: {problematic_vector}")
    logger.info(f"Vector string format: {str(problematic_vector)}")
    
    # This should work but currently fails with SQLCODE: <-104>
    result = insert_vector(
        cursor=cursor,
        table_name=TEST_TABLE_NAME,
        vector_column_name=TEST_VECTOR_COLUMN,
        vector_data=problematic_vector,
        target_dimension=TEST_DIMENSION,
        key_columns={"test_id": "reconciliation_repro"},
        additional_data={"test_name": "reconciliation_reproduction"}
    )
    
    assert result is True, "Reconciliation-style negative vector should succeed"
    real_iris_connection.commit()
    
    # Verify the vector was inserted correctly
    cursor.execute(f"SELECT {TEST_VECTOR_COLUMN} FROM {TEST_TABLE_NAME} WHERE test_id = ?", ("reconciliation_repro",))
    row = cursor.fetchone()
    assert row is not None, "Vector should be retrievable after insertion"
    
    # Verify the vector values are preserved correctly
    vector_str = row[0]
    assert vector_str is not None, "Vector should not be NULL"
    logger.info(f"Retrieved vector string: {vector_str}")