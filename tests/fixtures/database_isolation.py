"""
Database Isolation Fixtures
==========================

Provides test isolation for IRIS database to prevent state contamination
between tests and support MCP integration testing.
"""

import pytest
import logging
import uuid
from typing import Optional, Dict, Any
from contextlib import contextmanager

from tests.test_modes import MockController, TestMode

logger = logging.getLogger(__name__)

# Test namespace configuration
TEST_NAMESPACES = {
    TestMode.UNIT: "USER",  # Use default for unit tests with mocks
    TestMode.INTEGRATION: "RAG_TEST_INT",
    TestMode.E2E: "RAG_TEST_E2E"
}

# Track current test run
_current_test_run_id = None


def get_test_run_id() -> str:
    """Get or create a unique test run ID."""
    global _current_test_run_id
    if _current_test_run_id is None:
        _current_test_run_id = str(uuid.uuid4())[:8]
    return _current_test_run_id


def get_test_namespace() -> str:
    """Get the appropriate namespace for current test mode."""
    mode = MockController.get_test_mode()
    return TEST_NAMESPACES.get(mode, "USER")


def get_test_table_name(base_name: str, use_prefix: bool = True) -> str:
    """
    Get table name with test prefix if needed.
    
    Args:
        base_name: Base table name (e.g., "SourceDocuments")
        use_prefix: Whether to add test run prefix
        
    Returns:
        Fully qualified table name
    """
    namespace = get_test_namespace()
    
    if use_prefix and MockController.require_real_database():
        prefix = f"TEST_{get_test_run_id()}_"
        return f"{namespace}.{prefix}{base_name}"
    else:
        return f"{namespace}.{base_name}"


@pytest.fixture(scope="function")
def isolated_database(request):
    """
    Provides an isolated database environment for each test.
    
    Features:
    - Cleans up before and after test
    - Uses test-specific table names
    - Preserves data if marked with @pytest.mark.preserve_data
    - Skips cleanup for mock-only tests
    """
    if not MockController.require_real_database():
        # Just provide mock info for unit tests
        yield {
            "namespace": "MOCK",
            "table_prefix": "MOCK_",
            "cleanup_required": False
        }
        return
    
    # Get connection
    from common.iris_connection_manager import get_iris_connection
    conn = get_iris_connection()
    
    if not conn:
        pytest.skip("No database connection available")
        return
    
    # Setup test environment
    test_info = {
        "namespace": get_test_namespace(),
        "table_prefix": f"TEST_{get_test_run_id()}_",
        "cleanup_required": True
    }
    
    # Pre-test cleanup
    _cleanup_test_tables(conn, test_info["table_prefix"])
    
    # Create test tables if needed
    _ensure_test_tables(conn, test_info["table_prefix"])
    
    yield test_info
    
    # Post-test cleanup
    if request.node.get_closest_marker("preserve_data"):
        logger.info(f"Preserving test data in {test_info['namespace']}")
    else:
        _cleanup_test_tables(conn, test_info["table_prefix"])
    
    conn.close()


def _cleanup_test_tables(conn, table_prefix: str):
    """Clean up test-specific tables."""
    tables = [
        f"{table_prefix}SourceDocuments",
        f"{table_prefix}DocumentChunks",
        f"{table_prefix}DocumentTokenEmbeddings"
    ]
    
    namespace = get_test_namespace()
    cursor = conn.cursor()
    
    for table in tables:
        try:
            # Check if table exists first
            cursor.execute(
                f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
                f"WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
                (namespace, table)
            )
            
            if cursor.fetchone()[0] > 0:
                cursor.execute(f"DELETE FROM {namespace}.{table}")
                logger.debug(f"Cleaned table {namespace}.{table}")
        except Exception as e:
            logger.warning(f"Failed to clean {namespace}.{table}: {e}")
    
    conn.commit()
    cursor.close()


def _ensure_test_tables(conn, table_prefix: str):
    """Ensure test tables exist with proper schema."""
    namespace = get_test_namespace()
    cursor = conn.cursor()
    
    # Define table schemas matching production
    table_definitions = {
        f"{table_prefix}SourceDocuments": """
            CREATE TABLE IF NOT EXISTS {namespace}.{table_name} (
                id VARCHAR(255) PRIMARY KEY,
                content CLOB,
                metadata VARCHAR(4096),
                embedding VECTOR(DOUBLE, 1536),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        f"{table_prefix}DocumentChunks": """
            CREATE TABLE IF NOT EXISTS {namespace}.{table_name} (
                id VARCHAR(255) PRIMARY KEY,
                source_doc_id VARCHAR(255),
                content CLOB,
                chunk_index INTEGER,
                embedding VECTOR(DOUBLE, 1536),
                metadata VARCHAR(4096)
            )
        """,
        f"{table_prefix}DocumentTokenEmbeddings": """
            CREATE TABLE IF NOT EXISTS {namespace}.{table_name} (
                id VARCHAR(255) PRIMARY KEY,
                chunk_id VARCHAR(255),
                token_index INTEGER,
                token VARCHAR(512),
                embedding VECTOR(DOUBLE, 384)
            )
        """
    }
    
    for table_name, schema in table_definitions.items():
        try:
            sql = schema.format(namespace=namespace, table_name=table_name)
            cursor.execute(sql)
            logger.debug(f"Ensured table {namespace}.{table_name} exists")
        except Exception as e:
            logger.error(f"Failed to create {namespace}.{table_name}: {e}")
            raise
    
    conn.commit()
    cursor.close()


@pytest.fixture
def verify_clean_state():
    """Fixture to verify database is in clean state."""
    def _verify():
        if not MockController.require_real_database():
            return True
        
        from common.iris_connection_manager import get_iris_connection
        conn = get_iris_connection()
        
        if not conn:
            return False
        
        cursor = conn.cursor()
        namespace = get_test_namespace()
        prefix = f"TEST_{get_test_run_id()}_"
        
        tables = ["SourceDocuments", "DocumentChunks", "DocumentTokenEmbeddings"]
        
        for table in tables:
            full_name = f"{namespace}.{prefix}{table}"
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {full_name}")
                count = cursor.fetchone()[0]
                if count > 0:
                    logger.warning(f"Table {full_name} not clean: {count} rows")
                    return False
            except:
                # Table doesn't exist, which is fine
                pass
        
        cursor.close()
        conn.close()
        return True
    
    return _verify


@contextmanager
def temporary_test_data(docs: list):
    """
    Context manager for temporary test data.
    
    Usage:
        with temporary_test_data([doc1, doc2]) as table_info:
            # Run tests with data
            pipeline.query("test")
        # Data automatically cleaned up
    """
    if not MockController.require_real_database():
        yield {"table_prefix": "MOCK_"}
        return
    
    from common.iris_connection_manager import get_iris_connection
    conn = get_iris_connection()
    
    if not conn:
        raise RuntimeError("No database connection")
    
    table_info = {
        "namespace": get_test_namespace(),
        "table_prefix": f"TEMP_{uuid.uuid4().hex[:8]}_"
    }
    
    try:
        # Create temporary tables
        _ensure_test_tables(conn, table_info["table_prefix"])
        
        # Insert test data
        # ... (implementation depends on document format)
        
        yield table_info
        
    finally:
        # Always clean up
        _cleanup_test_tables(conn, table_info["table_prefix"])
        conn.close()


# MCP-specific fixtures

@pytest.fixture
async def mcp_test_environment():
    """
    Provides isolated environment for MCP integration testing.
    
    This fixture:
    - Creates a dedicated namespace for MCP tests
    - Ensures Python and Node.js see same data
    - Provides cleanup after tests
    """
    from tests.utils.mcp_test_helpers import MCPTestEnvironment
    
    env = MCPTestEnvironment()
    await env.setup()
    
    yield env
    
    await env.teardown()


@pytest.fixture
def assert_database_state():
    """
    Fixture for asserting expected database state.
    
    Usage:
        def test_something(assert_database_state):
            # Do something
            assert_database_state(docs=5, chunks=20)
    """
    def _assert(docs=0, chunks=0, embeddings=0):
        if not MockController.require_real_database():
            logger.info("Skipping database state assertion in mock mode")
            return
        
        from common.iris_connection_manager import get_iris_connection
        conn = get_iris_connection()
        
        cursor = conn.cursor()
        namespace = get_test_namespace()
        prefix = f"TEST_{get_test_run_id()}_"
        
        # Check documents
        cursor.execute(f"SELECT COUNT(*) FROM {namespace}.{prefix}SourceDocuments")
        actual_docs = cursor.fetchone()[0]
        assert actual_docs == docs, f"Expected {docs} documents, found {actual_docs}"
        
        # Check chunks
        cursor.execute(f"SELECT COUNT(*) FROM {namespace}.{prefix}DocumentChunks")
        actual_chunks = cursor.fetchone()[0]
        assert actual_chunks == chunks, f"Expected {chunks} chunks, found {actual_chunks}"
        
        # Check embeddings if specified
        if embeddings > 0:
            cursor.execute(f"SELECT COUNT(*) FROM {namespace}.{prefix}DocumentTokenEmbeddings")
            actual_embeddings = cursor.fetchone()[0]
            assert actual_embeddings == embeddings, f"Expected {embeddings} embeddings, found {actual_embeddings}"
        
        cursor.close()
        conn.close()
    
    return _assert