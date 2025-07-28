"""
Test DBAPI connection to licensed IRIS container.

This module tests the direct DBAPI connection to the IRIS database
using the intersystems-iris package.
"""

import pytest
import os

@pytest.fixture(scope="module", autouse=True)
def set_compose_file():
    """Set COMPOSE_FILE environment variable for this test module."""
    os.environ["COMPOSE_FILE"] = "docker-compose.licensed.yml"
    yield
    del os.environ["COMPOSE_FILE"]


@pytest.mark.skip(reason="Skipping due to persistent circular import issue in intersystems_iris library")
def test_dbapi_connection_to_licensed_iris(iris_container):
    """
    Test that the DBAPI can successfully connect to the licensed IRIS container.
    
    This test verifies:
    1. The licensed IRIS container is running (via iris_container fixture)
    2. The DBAPI can establish a connection
    3. The connection is functional
    
    Args:
        iris_container: Pytest fixture that ensures the licensed IRIS container is running
                       and returns connection parameters as a dict
    """
    import intersystems_iris.dbapi as iris
    
    # Attempt to connect using DBAPI with container parameters
    connection = None
    try:
        connection = iris.connect(**iris_container)
        
        # Assert that connection is successful
        assert connection is not None, "DBAPI connection should not be None"
        
        # Verify connection is functional by executing a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT 1 as test_value")
        result = cursor.fetchone()
        
        # Assert that we got a result
        assert result is not None, "Query result should not be None"
        assert result[0] == 1, "Query should return the expected value"
        
        cursor.close()
        
    except Exception as e:
        pytest.fail(f"DBAPI connection failed: {str(e)}")
        
    finally:
        # Clean up connection
        if connection:
            connection.close()


@pytest.mark.skip(reason="Skipping due to persistent circular import issue in intersystems_iris library")
def test_dbapi_connection_parameters_validation(iris_container):
    """
    Test that DBAPI connection fails appropriately with invalid parameters.
    
    This test verifies that the DBAPI properly handles invalid connection parameters.
    
    Args:
        iris_container: Pytest fixture that ensures the licensed IRIS container is running
    """
    import intersystems_iris.dbapi as iris
    
    # Test with invalid port
    invalid_params = iris_container.copy()
    invalid_params['port'] = 9999  # Invalid port
    
    with pytest.raises(Exception):
        iris.connect(**invalid_params)
    
    # Test with invalid credentials
    invalid_creds = iris_container.copy()
    invalid_creds.update({
        'username': 'invalid_user',
        'password': 'invalid_password'
    })
    
    with pytest.raises(Exception):
        iris.connect(**invalid_creds)


@pytest.mark.skip(reason="Skipping due to persistent circular import issue in intersystems_iris library")
def test_dbapi_connection_basic_operations(iris_container):
    """
    Test basic database operations through DBAPI connection.
    
    This test verifies that basic SQL operations work through the DBAPI connection.
    
    Args:
        iris_container: Pytest fixture that ensures the licensed IRIS container is running
    """
    import intersystems_iris.dbapi as iris
    
    connection = None
    try:
        # Establish connection
        connection = iris.connect(**iris_container)
        
        cursor = connection.cursor()
        
        # Test CREATE TABLE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_dbapi_table (
                id INTEGER,
                name VARCHAR(50)
            )
        """)
        
        # Test INSERT
        cursor.execute("""
            INSERT INTO test_dbapi_table (id, name) VALUES (?, ?)
        """, (1, "test_record"))
        
        # Test SELECT
        cursor.execute("SELECT id, name FROM test_dbapi_table WHERE id = ?", (1,))
        result = cursor.fetchone()
        
        # Assert results
        assert result is not None, "Should retrieve the inserted record"
        assert result[0] == 1, "ID should match inserted value"
        assert result[1] == "test_record", "Name should match inserted value"
        
        # Test DELETE
        cursor.execute("DELETE FROM test_dbapi_table WHERE id = ?", (1,))
        
        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM test_dbapi_table WHERE id = ?", (1,))
        count_result = cursor.fetchone()
        assert count_result[0] == 0, "Record should be deleted"
        
        # Clean up table
        cursor.execute("DROP TABLE test_dbapi_table")
        
        cursor.close()
        
    except Exception as e:
        pytest.fail(f"DBAPI basic operations failed: {str(e)}")
        
    finally:
        if connection:
            connection.close()


@pytest.mark.skip(reason="Skipping due to persistent circular import issue in intersystems_iris library")
def test_dbapi_connection_iris_specific_features(iris_container):
    """
    Test IRIS-specific features through DBAPI connection.
    
    This test verifies that IRIS-specific SQL features work through the DBAPI.
    
    Args:
        iris_container: Pytest fixture that ensures the licensed IRIS container is running
    """
    import intersystems_iris.dbapi as iris
    
    connection = None
    try:
        # Establish connection
        connection = iris.connect(**iris_container)
        
        cursor = connection.cursor()
        
        # Test IRIS-specific TOP syntax (instead of LIMIT)
        cursor.execute("SELECT TOP 5 1 as test_value")
        results = cursor.fetchall()
        
        # Assert we got exactly 5 results
        assert len(results) == 5, "Should return exactly 5 rows with TOP 5"
        assert all(row[0] == 1 for row in results), "All rows should have test_value = 1"
        
        # Test IRIS system functions
        cursor.execute("SELECT $HOROLOG")
        horolog_result = cursor.fetchone()
        assert horolog_result is not None, "Should get $HOROLOG system variable"
        assert horolog_result[0] is not None, "$HOROLOG should not be None"
        
        cursor.close()
        
    except Exception as e:
        pytest.fail(f"IRIS-specific DBAPI features failed: {str(e)}")
        
    finally:
        if connection:
            connection.close()


@pytest.mark.skip(reason="Skipping due to persistent circular import issue in intersystems_iris library")
def test_dbapi_connection_transaction_support(iris_container):
    """
    Test transaction support through DBAPI connection.
    
    This test verifies that transaction operations work correctly.
    
    Args:
        iris_container: Pytest fixture that ensures the licensed IRIS container is running
    """
    import intersystems_iris.dbapi as iris
    
    connection = None
    try:
        # Establish connection
        connection = iris.connect(**iris_container)
        
        cursor = connection.cursor()
        
        # Create test table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_transaction_table (
                id INTEGER,
                value VARCHAR(50)
            )
        """)
        
        # Test transaction rollback
        cursor.execute("INSERT INTO test_transaction_table (id, value) VALUES (?, ?)", (1, "before_rollback"))
        connection.rollback()  # Rollback the insert
        
        # Verify rollback worked
        cursor.execute("SELECT COUNT(*) FROM test_transaction_table WHERE id = ?", (1,))
        count_after_rollback = cursor.fetchone()[0]
        assert count_after_rollback == 0, "Record should not exist after rollback"
        
        # Test transaction commit
        cursor.execute("INSERT INTO test_transaction_table (id, value) VALUES (?, ?)", (2, "after_commit"))
        connection.commit()  # Commit the insert
        
        # Verify commit worked
        cursor.execute("SELECT COUNT(*) FROM test_transaction_table WHERE id = ?", (2,))
        count_after_commit = cursor.fetchone()[0]
        assert count_after_commit == 1, "Record should exist after commit"
        
        # Clean up
        cursor.execute("DROP TABLE test_transaction_table")
        connection.commit()
        
        cursor.close()
        
    except Exception as e:
        pytest.fail(f"DBAPI transaction support failed: {str(e)}")
        
    finally:
        if connection:
            connection.close()