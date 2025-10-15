"""Integration test configuration and fixtures.

This module provides IRIS database fixtures and real service configurations
for integration tests. Integration tests use real IRIS connections but may
mock external APIs (LLM, embeddings).
"""

import pytest
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock


@pytest.fixture(scope="session")
def iris_connection_config():
    """IRIS database connection configuration for integration tests.

    Uses port discovery to find available IRIS instance:
    - 31972: Test database (docker-compose.test.yml)
    - 1972: System/production database
    - Other configured ports
    """
    test_ports = [31972, 1972, 11972, 21972]

    for port in test_ports:
        try:
            # Test connection
            result = subprocess.run([
                sys.executable, "-c",
                f"""
import sqlalchemy_iris
from sqlalchemy import create_engine, text
try:
    engine = create_engine(f'iris://_SYSTEM:SYS@localhost:{port}/USER')
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    print('SUCCESS')
except Exception:
    print('FAILED')
"""
            ], capture_output=True, text=True, timeout=5)

            if "SUCCESS" in result.stdout:
                return {
                    "host": "localhost",
                    "port": port,
                    "username": "_SYSTEM",
                    "password": "SYS",
                    "namespace": "USER",
                    "connection_string": f"iris://_SYSTEM:SYS@localhost:{port}/USER"
                }
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            continue

    # No IRIS instance found - skip integration tests
    pytest.skip("No IRIS database available for integration tests")


@pytest.fixture(scope="session")
def iris_engine(iris_connection_config):
    """Create SQLAlchemy engine for IRIS database."""
    try:
        import sqlalchemy_iris
        from sqlalchemy import create_engine

        engine = create_engine(iris_connection_config["connection_string"])

        # Test connection
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))

        yield engine

        # Cleanup
        engine.dispose()
    except Exception as e:
        pytest.skip(f"Could not create IRIS engine: {e}")


@pytest.fixture
def iris_connection(iris_engine):
    """Provide IRIS database connection for integration tests."""
    conn = iris_engine.connect()

    try:
        yield conn
    finally:
        # Rollback any uncommitted changes
        try:
            conn.rollback()
        except:
            pass
        conn.close()


@pytest.fixture
def iris_test_table(iris_connection):
    """Create and cleanup test table for integration tests."""
    from sqlalchemy import text

    table_name = "integration_test_vectors"

    # Create test table
    try:
        iris_connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR(255) PRIMARY KEY,
                content CLOB,
                metadata VARCHAR(2000),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        iris_connection.commit()
    except Exception as e:
        print(f"Warning: Could not create test table: {e}")

    yield table_name

    # Cleanup: Drop test table
    try:
        iris_connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        iris_connection.commit()
    except Exception as e:
        print(f"Warning: Could not drop test table: {e}")


@pytest.fixture
def iris_vector_store_config(iris_connection_config):
    """Configuration for IRIS vector store integration tests."""
    return {
        "connection_string": iris_connection_config["connection_string"],
        "table_name": "integration_test_vectors",
        "vector_dimension": 384,
        "distance_metric": "COSINE"
    }


@pytest.fixture
def integration_test_config(iris_connection_config) -> Dict[str, Any]:
    """Full configuration for integration tests with real IRIS connection."""
    return {
        "database": {
            "iris": {
                "host": iris_connection_config["host"],
                "port": iris_connection_config["port"],
                "username": iris_connection_config["username"],
                "password": iris_connection_config["password"],
                "namespace": iris_connection_config["namespace"]
            }
        },
        "llm": {
            "provider": "mock",  # Use mock for LLM in integration tests
            "model": "test-model"
        },
        "embeddings": {
            "provider": "mock",  # Use mock for embeddings in integration tests
            "model": "test-embedding-model",
            "dimension": 384
        },
        "vector_store": {
            "provider": "iris",
            "table_name": "integration_test_vectors"
        }
    }


@pytest.fixture
def mock_llm_for_integration():
    """Mock LLM client for integration tests (avoid real API calls)."""
    mock_llm = Mock()
    mock_llm.generate = Mock(return_value="Integration test response")
    mock_llm.chat = Mock(return_value="Integration test chat response")
    mock_llm.__call__ = Mock(return_value="Integration test LLM response")
    return mock_llm


@pytest.fixture
def mock_embeddings_for_integration():
    """Mock embedding service for integration tests (avoid model downloads)."""
    mock_embeddings = Mock()
    mock_embeddings.embed_documents = Mock(return_value=[[0.1] * 384 for _ in range(10)])
    mock_embeddings.embed_query = Mock(return_value=[0.1] * 384)
    mock_embeddings.dimension = 384
    return mock_embeddings


@pytest.fixture
def sample_documents_for_integration():
    """Sample documents for integration testing."""
    return [
        {
            "id": "int_doc1",
            "content": "Integration test document about IRIS database capabilities.",
            "metadata": {"source": "integration_test", "type": "database"}
        },
        {
            "id": "int_doc2",
            "content": "RAG pipeline integration with vector search functionality.",
            "metadata": {"source": "integration_test", "type": "pipeline"}
        },
        {
            "id": "int_doc3",
            "content": "Entity extraction and relationship mapping in knowledge graphs.",
            "metadata": {"source": "integration_test", "type": "graph"}
        }
    ]


@pytest.fixture(autouse=True)
def cleanup_test_data(iris_connection):
    """Cleanup test data after each integration test."""
    yield

    # Cleanup test tables
    from sqlalchemy import text
    test_tables = [
        "integration_test_vectors",
        "test_documents",
        "test_entities",
        "test_relationships"
    ]

    for table in test_tables:
        try:
            iris_connection.execute(text(f"DROP TABLE IF EXISTS {table}"))
            iris_connection.commit()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
def iris_schema_manager(iris_connection):
    """Schema manager for integration tests."""
    from iris_rag.storage.schema_manager import SchemaManager
    from unittest.mock import Mock

    # Create schema manager with real connection
    mock_config = Mock()
    mock_config.get = Mock(side_effect=lambda key, default=None: {
        "database.iris.host": "localhost",
        "database.iris.port": 31972,
        "database.iris.username": "_SYSTEM",
        "database.iris.password": "SYS",
        "database.iris.namespace": "USER",
    }.get(key, default))

    try:
        manager = SchemaManager(config=mock_config)
        manager.connection = iris_connection  # Use test connection
        yield manager
    except Exception as e:
        pytest.skip(f"Could not create SchemaManager: {e}")


@pytest.fixture(scope="session", autouse=True)
def verify_iris_available():
    """Verify IRIS is available before running integration tests."""
    test_ports = [31972, 1972, 11972]

    for port in test_ports:
        try:
            result = subprocess.run([
                sys.executable, "-c",
                f"""
import sqlalchemy_iris
from sqlalchemy import create_engine, text
try:
    engine = create_engine(f'iris://_SYSTEM:SYS@localhost:{port}/USER')
    with engine.connect() as conn:
        conn.execute(text('SELECT 1'))
    print('AVAILABLE')
except Exception:
    pass
"""
            ], capture_output=True, text=True, timeout=5)

            if "AVAILABLE" in result.stdout:
                return  # IRIS is available
        except:
            continue

    # No IRIS available - warn but don't fail session
    print("\nWARNING: No IRIS database available. Integration tests will be skipped.\n")
