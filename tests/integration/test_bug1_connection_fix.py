"""
Integration tests for Bug 1: Connection API Fix (v0.5.4)

Tests verify that:
1. intersystems_iris.connect() is used correctly (official intersystems-irispython v5.3.0)
2. No AttributeError during connection establishment
3. ConnectionError raised (not AttributeError) for invalid connections
4. Error messages are clear and actionable

Note: The PyPI documentation says 'import iris' but in environments with multiple
iris-related packages, we must use 'import intersystems_iris' explicitly.
"""

import os
import pytest
from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection


class TestBug1ConnectionAPIFix:
    """Integration tests for Bug 1: iris.createConnection() fix"""

    def test_connection_establishment_success(self):
        """T022: Verify connection establishes successfully without AttributeError"""
        # Given: Valid IRIS connection parameters (from environment)
        # When: get_iris_dbapi_connection() is called
        conn = get_iris_dbapi_connection()

        # Then: Connection succeeds without AttributeError
        assert conn is not None, "Connection should not be None"

        # And: Connection is usable (test query)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        assert result is not None, "Test query should return result"
        print("✅ Bug 1 Fix: Connection established successfully using intersystems_iris.connect()")

    def test_no_attribute_error_on_connection(self):
        """T022: Verify no AttributeError about missing 'connect' method"""
        # Given: Valid IRIS connection parameters
        # When: get_iris_dbapi_connection() is called
        # Then: No AttributeError is raised
        try:
            conn = get_iris_dbapi_connection()
            assert conn is not None
            conn.close()
            print("✅ Bug 1 Fix: No AttributeError during connection")
        except AttributeError as e:
            pytest.fail(f"AttributeError should not occur: {e}")

    def test_connection_error_for_invalid_port(self):
        """T023: Verify ConnectionError (not AttributeError) for invalid port"""
        # Given: Invalid IRIS connection parameters (wrong port)
        original_port = os.environ.get("IRIS_PORT")
        os.environ["IRIS_PORT"] = "9999"  # Invalid port

        try:
            # When: get_iris_dbapi_connection() is called
            # Then: ConnectionError raised (not AttributeError)
            with pytest.raises(ConnectionError) as exc_info:
                get_iris_dbapi_connection()

            # And: Error message is clear and actionable
            error_msg = str(exc_info.value)
            assert "Failed to connect" in error_msg
            assert "9999" in error_msg  # Port mentioned
            print(f"✅ Bug 1 Fix: ConnectionError raised with clear message: {error_msg[:100]}...")

        finally:
            # Restore original port
            if original_port:
                os.environ["IRIS_PORT"] = original_port
            else:
                os.environ.pop("IRIS_PORT", None)

    def test_error_message_clarity(self):
        """T023: Verify error messages follow standard format"""
        # Given: Invalid IRIS connection parameters (wrong port)
        original_port = os.environ.get("IRIS_PORT")
        os.environ["IRIS_PORT"] = "9999"

        try:
            # When: get_iris_dbapi_connection() is called
            with pytest.raises(ConnectionError) as exc_info:
                get_iris_dbapi_connection()

            # Then: Error message contains expected elements
            error_msg = str(exc_info.value)
            assert "Failed to connect" in error_msg or "connection failed" in error_msg.lower()
            assert "9999" in error_msg  # Port mentioned

            # And: No mention of AttributeError or iris.connect()
            assert "AttributeError" not in error_msg
            assert "iris.connect" not in error_msg

            print("✅ Bug 1 Fix: Error message format is clear and actionable")

        finally:
            if original_port:
                os.environ["IRIS_PORT"] = original_port
            else:
                os.environ.pop("IRIS_PORT", None)

    def test_connection_manager_integration(self):
        """T022: Verify ConnectionManager works after line 210 fix"""
        from iris_vector_rag.common.connection_manager import ConnectionManager

        # Given: ConnectionManager initialized with dbapi connection type
        connection_manager = ConnectionManager(connection_type="dbapi")

        # When: Connection is established
        conn = connection_manager.connect()

        # Then: Connection succeeds without AttributeError
        assert conn is not None

        # And: Connection is usable
        with connection_manager.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result is not None

        connection_manager.close()
        print("✅ Bug 1 Fix: ConnectionManager integration successful")

    def test_multiple_sequential_connections(self):
        """T022: Verify multiple connections work without errors"""
        # Test that we can create multiple connections sequentially
        # This verifies that the fix doesn't have issues with connection lifecycle

        for i in range(3):
            conn = get_iris_dbapi_connection()
            assert conn is not None, f"Connection {i+1} should not be None"

            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            assert result is not None, f"Test query {i+1} should return result"

        print("✅ Bug 1 Fix: Multiple sequential connections successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
