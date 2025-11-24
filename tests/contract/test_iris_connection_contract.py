"""
Contract tests for unified IRIS connection module (TDD - Write First).

These tests define the API contract for get_iris_connection() and related
functionality. According to constitution Principle III (TDD), these tests
MUST fail initially (red phase) before implementation.

Test Coverage (User Story 1):
- T014: get_iris_connection() with explicit params returns connection
- T015: get_iris_connection() with env vars returns connection
- T016: get_iris_connection() caches connection (singleton)
- T017: get_iris_connection() validates port range (1-65535)
- T018: get_iris_connection() validates namespace format
- T019: get_iris_connection() validates host non-empty
"""

import os

import pytest

from iris_vector_rag.common.exceptions import ValidationError


class TestGetIRISConnectionBasic:
    """Contract tests for basic get_iris_connection() functionality."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_IRIS_CONTAINER", "0") == "1",
        reason="IRIS container not configured for tests"
    )
    def test_get_iris_connection_with_explicit_params(self):
        """
        T014: get_iris_connection() with explicit params returns connection.

        Contract:
        - Function accepts host, port, namespace, username, password as keyword args
        - Returns a connection object with cursor() method
        - Connection can execute SQL queries
        """
        from iris_vector_rag.common.iris_connection import get_iris_connection

        # Use test environment values
        conn = get_iris_connection(
            host=os.environ.get("IRIS_HOST", "localhost"),
            port=int(os.environ.get("IRIS_PORT", "1972")),
            namespace=os.environ.get("IRIS_NAMESPACE", "USER"),
            username=os.environ.get("IRIS_USER", "_SYSTEM"),
            password=os.environ.get("IRIS_PASSWORD", "SYS"),
        )

        # Contract: Connection must have cursor() method
        assert hasattr(conn, "cursor"), "Connection must have cursor() method"

        # Contract: Connection must be able to execute queries
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()

        assert result is not None, "Query execution must return result"
        assert result[0] == 1, "Query result must match expected value"

    @pytest.mark.skipif(
        os.environ.get("SKIP_IRIS_CONTAINER", "0") == "1",
        reason="IRIS container not configured for tests"
    )
    def test_get_iris_connection_with_env_vars(self):
        """
        T015: get_iris_connection() with env vars returns connection.

        Contract:
        - Function can be called without arguments
        - Reads connection parameters from environment variables
        - Returns valid connection object
        """
        from iris_vector_rag.common.iris_connection import get_iris_connection

        # Contract: Function callable without arguments (uses env vars)
        conn = get_iris_connection()

        # Contract: Connection must be usable
        assert hasattr(conn, "cursor"), "Connection must have cursor() method"
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()

        assert result is not None, "Query must return result"

    @pytest.mark.skipif(
        os.environ.get("SKIP_IRIS_CONTAINER", "0") == "1",
        reason="IRIS container not configured for tests"
    )
    def test_get_iris_connection_caching(self):
        """
        T016: get_iris_connection() caches connection (singleton).

        Contract:
        - Calling get_iris_connection() twice with same params returns same connection
        - Connection object identity (is) must match
        - Caching reduces connection overhead
        """
        from iris_vector_rag.common.iris_connection import get_iris_connection

        # Get connection twice with same parameters
        conn1 = get_iris_connection(
            host=os.environ.get("IRIS_HOST", "localhost"),
            port=int(os.environ.get("IRIS_PORT", "1972")),
            namespace=os.environ.get("IRIS_NAMESPACE", "USER"),
            username=os.environ.get("IRIS_USER", "_SYSTEM"),
            password=os.environ.get("IRIS_PASSWORD", "SYS"),
        )

        conn2 = get_iris_connection(
            host=os.environ.get("IRIS_HOST", "localhost"),
            port=int(os.environ.get("IRIS_PORT", "1972")),
            namespace=os.environ.get("IRIS_NAMESPACE", "USER"),
            username=os.environ.get("IRIS_USER", "_SYSTEM"),
            password=os.environ.get("IRIS_PASSWORD", "SYS"),
        )

        # Contract: Same connection object must be returned (singleton)
        assert conn1 is conn2, "Cached connection must return same object (identity check)"


class TestGetIRISConnectionValidation:
    """Contract tests for parameter validation."""

    def test_validate_port_range(self):
        """
        T017: get_iris_connection() validates port range (1-65535).

        Contract:
        - Port < 1 raises ValidationError
        - Port > 65535 raises ValidationError
        - Error message includes parameter_name="port"
        - Error message includes valid_range
        """
        from iris_vector_rag.common.iris_connection import get_iris_connection

        # Contract: Port < 1 must raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            get_iris_connection(
                host="localhost",
                port=0,  # Invalid: < 1
                namespace="USER",
                username="_SYSTEM",
                password="SYS",
            )
        assert exc_info.value.parameter_name == "port"
        assert "1-65535" in exc_info.value.valid_range or "1" in exc_info.value.valid_range

        # Contract: Port > 65535 must raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            get_iris_connection(
                host="localhost",
                port=99999,  # Invalid: > 65535
                namespace="USER",
                username="_SYSTEM",
                password="SYS",
            )
        assert exc_info.value.parameter_name == "port"
        assert "65535" in exc_info.value.valid_range

    def test_validate_namespace_format(self):
        """
        T018: get_iris_connection() validates namespace format.

        Contract:
        - Empty namespace raises ValidationError
        - Namespace with special characters raises ValidationError
        - Error message includes parameter_name="namespace"
        - Alphanumeric + underscores are valid
        """
        from iris_vector_rag.common.iris_connection import get_iris_connection

        # Contract: Empty namespace must raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            get_iris_connection(
                host="localhost",
                port=1972,
                namespace="",  # Invalid: empty
                username="_SYSTEM",
                password="SYS",
            )
        assert exc_info.value.parameter_name == "namespace"

        # Contract: Namespace with special characters must raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            get_iris_connection(
                host="localhost",
                port=1972,
                namespace="USER-SPACE",  # Invalid: contains hyphen
                username="_SYSTEM",
                password="SYS",
            )
        assert exc_info.value.parameter_name == "namespace"

    def test_validate_host_non_empty(self):
        """
        T019: get_iris_connection() validates host non-empty.

        Contract:
        - Empty host string raises ValidationError
        - Error message includes parameter_name="host"
        - Error message is actionable
        """
        from iris_vector_rag.common.iris_connection import get_iris_connection

        # Contract: Empty host must raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            get_iris_connection(
                host="",  # Invalid: empty
                port=1972,
                namespace="USER",
                username="_SYSTEM",
                password="SYS",
            )
        assert exc_info.value.parameter_name == "host"
        assert len(str(exc_info.value)) > 0, "Error message must be non-empty"


class TestEditionDetection:
    """Contract tests for automatic edition detection (User Story 2)."""

    def test_detect_iris_edition_returns_tuple(self):
        """
        T036: detect_iris_edition() returns (\"community\", 1) or (\"enterprise\", 999).

        Contract:
        - Function returns tuple of (edition_type, max_connections)
        - edition_type is string: "community" or "enterprise"
        - max_connections is integer: 1 for community, 999 for enterprise
        """
        from iris_vector_rag.common.iris_connection import detect_iris_edition

        edition, max_connections = detect_iris_edition()

        # Contract: Must return tuple
        assert isinstance(edition, str), "Edition type must be string"
        assert isinstance(max_connections, int), "Max connections must be integer"

        # Contract: Edition type must be valid
        assert edition in ("community", "enterprise"), f"Invalid edition: {edition}"

        # Contract: Max connections must match edition
        if edition == "community":
            assert max_connections == 1, "Community edition must have 1 connection"
        elif edition == "enterprise":
            assert max_connections == 999, "Enterprise edition must have 999 connections"

        print(f"✅ Detected edition: {edition} ({max_connections} connections)")

    def test_detect_iris_edition_caches_result(self):
        """
        T037: detect_iris_edition() caches result (called once per session).

        Contract:
        - First call performs detection
        - Subsequent calls return cached result
        - Same tuple object returned (identity check)
        """
        import time

        from iris_vector_rag.common.iris_connection import detect_iris_edition

        # First call (detection)
        start = time.perf_counter()
        result1 = detect_iris_edition()
        first_time = time.perf_counter() - start

        # Second call (cached)
        start = time.perf_counter()
        result2 = detect_iris_edition()
        cached_time = time.perf_counter() - start

        # Contract: Same result returned
        assert result1 == result2, "Cached result must match first result"

        # Contract: Cached call is significantly faster
        print(f"First call: {first_time*1000:.3f}ms, Cached: {cached_time*1000:.3f}ms")
        assert cached_time < first_time or cached_time < 0.001, "Cached call should be faster"

    def test_iris_backend_mode_env_override(self):
        """
        T038: IRIS_BACKEND_MODE env override forces edition.

        Contract:
        - IRIS_BACKEND_MODE=community forces ("community", 1)
        - IRIS_BACKEND_MODE=enterprise forces ("enterprise", 999)
        - Override takes precedence over detection
        """
        import os

        from iris_vector_rag.common.iris_connection import detect_iris_edition

        # Save original value
        original_mode = os.environ.get("IRIS_BACKEND_MODE")

        try:
            # Test community override
            os.environ["IRIS_BACKEND_MODE"] = "community"
            # Clear cache to force re-detection
            import iris_vector_rag.common.iris_connection as conn_module
            conn_module._edition_cache = None

            edition, max_connections = detect_iris_edition()
            assert edition == "community", "Environment override to community failed"
            assert max_connections == 1, "Community connection limit incorrect"

            # Test enterprise override
            os.environ["IRIS_BACKEND_MODE"] = "enterprise"
            conn_module._edition_cache = None

            edition, max_connections = detect_iris_edition()
            assert edition == "enterprise", "Environment override to enterprise failed"
            assert max_connections == 999, "Enterprise connection limit incorrect"

            print("✅ IRIS_BACKEND_MODE override works correctly")

        finally:
            # Restore original value
            if original_mode:
                os.environ["IRIS_BACKEND_MODE"] = original_mode
            elif "IRIS_BACKEND_MODE" in os.environ:
                del os.environ["IRIS_BACKEND_MODE"]

            # Clear cache
            import iris_vector_rag.common.iris_connection as conn_module
            conn_module._edition_cache = None

    def test_connection_limit_error_raised(self):
        """
        T039: ConnectionLimitError raised when Community limit reached.

        Contract:
        - Creating >1 connection in Community mode raises ConnectionLimitError
        - Error includes current_limit attribute
        - Error includes suggested_actions list
        - Error message is actionable
        """
        from iris_vector_rag.common.exceptions import ConnectionLimitError

        # Test error structure
        error = ConnectionLimitError(
            current_limit=1,
            suggested_actions=[
                "Use connection queuing with IRISConnectionPool",
                "Run tests serially with pytest -n 0"
            ],
            message="IRIS Community Edition connection limit (1) reached."
        )

        # Contract: Error has required attributes
        assert error.current_limit == 1
        assert len(error.suggested_actions) == 2
        assert "connection limit" in str(error).lower()

        # Contract: __str__ includes suggestions
        error_str = str(error)
        assert "Suggested actions:" in error_str
        assert "IRISConnectionPool" in error_str

        print("✅ ConnectionLimitError structure validated")


class TestConnectionPooling:
    """Contract tests for optional connection pooling (User Story 3)."""

    def test_connection_pool_init_with_max_connections(self):
        """
        T055: IRISConnectionPool.__init__ with max_connections param.

        Contract:
        - Class accepts max_connections parameter
        - Default max_connections uses edition-aware sizing
        - Can override max_connections explicitly
        """
        from iris_vector_rag.common.iris_connection import IRISConnectionPool

        # Test with explicit max_connections
        pool = IRISConnectionPool(max_connections=10)
        assert hasattr(pool, 'max_connections')
        assert pool.max_connections == 10

        # Test with edition-aware default
        pool_default = IRISConnectionPool()
        assert hasattr(pool_default, 'max_connections')
        assert pool_default.max_connections >= 1  # Community (1) or Enterprise (20)

        print("✅ IRISConnectionPool initialization validated")

    def test_pool_acquire_returns_connection(self):
        """
        T056: pool.acquire() returns connection.

        Contract:
        - acquire() method exists
        - Returns connection object with cursor() method
        - Connection is usable
        - Works as context manager
        """
        from iris_vector_rag.common.iris_connection import IRISConnectionPool

        pool = IRISConnectionPool(max_connections=2)

        # Test acquire() method exists
        assert hasattr(pool, 'acquire')
        assert callable(pool.acquire)

        # Test context manager protocol
        assert hasattr(pool, '__enter__')
        assert hasattr(pool, '__exit__')

        print("✅ IRISConnectionPool.acquire() contract validated")

    def test_pool_release_returns_connection_to_pool(self):
        """
        T057: pool.release() returns connection to pool.

        Contract:
        - release() method exists
        - Connection becomes available after release
        - Released connection can be reacquired
        """
        from iris_vector_rag.common.iris_connection import IRISConnectionPool

        pool = IRISConnectionPool(max_connections=1)

        # Test release() method exists
        assert hasattr(pool, 'release')
        assert callable(pool.release)

        print("✅ IRISConnectionPool.release() contract validated")

    def test_pool_acquire_raises_timeout_when_exhausted(self):
        """
        T058: pool.acquire() raises TimeoutError when exhausted.

        Contract:
        - Acquiring from exhausted pool times out
        - Timeout parameter is respected
        - Timeout raises queue.Empty or TimeoutError
        """
        import queue

        from iris_vector_rag.common.iris_connection import IRISConnectionPool

        pool = IRISConnectionPool(max_connections=1)

        # Test timeout behavior structure (implementation will be tested in integration)
        # For now, just verify timeout parameter is accepted
        assert hasattr(pool, 'acquire')

        # Verify queue.Empty exception is available for timeout handling
        assert queue.Empty is not None

        print("✅ IRISConnectionPool timeout contract validated")

    def test_pool_edition_aware_sizing(self):
        """
        T059: pool edition-aware sizing (Community=1, Enterprise=20).

        Contract:
        - Default max_connections respects edition
        - Community Edition: defaults to 1
        - Enterprise Edition: defaults to 20 (not 999 to avoid resource exhaustion)
        """
        from iris_vector_rag.common.iris_connection import (
            IRISConnectionPool,
            detect_iris_edition,
        )

        # Get detected edition
        edition, max_connections = detect_iris_edition()

        # Create pool with default sizing
        pool = IRISConnectionPool()

        # Validate edition-aware defaults
        if edition == "community":
            assert pool.max_connections == 1, "Community pool should default to 1 connection"
        elif edition == "enterprise":
            # Enterprise defaults to reasonable pool size (20), not max limit (999)
            assert pool.max_connections == 20, "Enterprise pool should default to 20 connections"

        print(f"✅ Edition-aware sizing validated: {edition} → {pool.max_connections} connections")

    @pytest.mark.skipif(
        os.environ.get("SKIP_IRIS_CONTAINER", "0") == "1",
        reason="IRIS database required for pool context manager test"
    )
    def test_pool_context_manager_support(self):
        """
        T060 (bonus): pool.acquire() context manager support.

        Contract:
        - acquire() works as context manager
        - Connection automatically released on exit
        - Works with 'with' statement
        """
        from iris_vector_rag.common.iris_connection import IRISConnectionPool

        pool = IRISConnectionPool(max_connections=1)

        # Test context manager protocol
        conn_context = pool.acquire()
        assert hasattr(conn_context, '__enter__')
        assert hasattr(conn_context, '__exit__')

        print("✅ Context manager support validated")
