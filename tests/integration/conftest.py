"""Integration test configuration and fixtures.

This module provides IRIS database fixtures and real service configurations
for integration tests. Integration tests use real IRIS connections but may
mock external APIs (LLM, embeddings).
"""

import sys
from typing import Any, Dict
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="session")
def iris_connection_config(iris_target):
    """Connection settings for the IRIS the session runs against (see tests/conftest.py)."""
    return {
        "host": iris_target.host,
        "port": iris_target.port,
        "username": iris_target.username,
        "password": iris_target.password,
        "namespace": iris_target.namespace,
        "connection_string": iris_target.connection_string,
    }


@pytest.fixture(scope="session")
def iris_engine(iris_connection_config):
    """SQLAlchemy engine for the same IRIS as `iris_connection`. Fails loudly, never skips."""
    from sqlalchemy import create_engine, text

    engine = create_engine(iris_connection_config["connection_string"])
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def medical_20_loaded(iris_target):
    """The `medical-graphrag-20` fixture (3 documents, 21 entities, 15 relationships).

    Loaded into RAG.* of the session's IRIS by FixtureManager (tables created in the
    canonical shape if absent, any rows already in them deleted first, checksum of the
    source data validated) and deleted again when the module is done, so the files that
    run after it start from empty tables rather than from this data.
    """
    from pathlib import Path

    from tests.fixtures.manager import FixtureManager
    from tests.fixtures.pytest_plugin import release_open_transactions
    from tests.support.iris_environment import connect

    connection = connect(iris_target)
    manager = FixtureManager(
        fixtures_root=Path(__file__).resolve().parents[1] / "fixtures",
        connection=connection,
    )
    try:
        # Earlier modules leave pipelines and their connections behind; an open
        # transaction among them would make the ALTER below wait for ever.
        release_open_transactions(keep=connection)
        result = manager.load_fixture("medical-graphrag-20", cleanup_first=True)
        assert result.success, result.error_message
        release_open_transactions(keep=connection)
        yield result
    finally:
        try:
            manager.cleanup_fixture("medical-graphrag-20")
        finally:
            connection.close()


@pytest.fixture
def iris_sql_connection(iris_engine):
    """SQLAlchemy connection for integration tests that use `text()` SQL."""
    conn = iris_engine.connect()

    try:
        yield conn
    finally:
        # Rollback any uncommitted changes
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()


@pytest.fixture
def iris_test_table(iris_sql_connection):
    """Create and cleanup test table for integration tests."""
    from sqlalchemy import text

    table_name = "integration_test_vectors"

    # Create test table
    try:
        iris_sql_connection.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR(255) PRIMARY KEY,
                content CLOB,
                metadata VARCHAR(2000),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        iris_sql_connection.commit()
    except Exception as e:
        print(f"Warning: Could not create test table: {e}")

    yield table_name

    # Cleanup: Drop test table
    try:
        iris_sql_connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        iris_sql_connection.commit()
    except Exception as e:
        print(f"Warning: Could not drop test table: {e}")


@pytest.fixture
def iris_vector_store_config(iris_connection_config):
    """Configuration for IRIS vector store integration tests."""
    return {
        "connection_string": iris_connection_config["connection_string"],
        "table_name": "integration_test_vectors",
        "vector_dimension": 384,
        "distance_metric": "COSINE",
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
                "namespace": iris_connection_config["namespace"],
            }
        },
        "llm": {
            "provider": "mock",  # Use mock for LLM in integration tests
            "model": "test-model",
        },
        "embeddings": {
            "provider": "mock",  # Use mock for embeddings in integration tests
            "model": "test-embedding-model",
            "dimension": 384,
        },
        "vector_store": {"provider": "iris", "table_name": "integration_test_vectors"},
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
    mock_embeddings.embed_documents = Mock(
        return_value=[[0.1] * 384 for _ in range(10)]
    )
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
            "metadata": {"source": "integration_test", "type": "database"},
        },
        {
            "id": "int_doc2",
            "content": "RAG pipeline integration with vector search functionality.",
            "metadata": {"source": "integration_test", "type": "pipeline"},
        },
        {
            "id": "int_doc3",
            "content": "Entity extraction and relationship mapping in knowledge graphs.",
            "metadata": {"source": "integration_test", "type": "graph"},
        },
    ]


@pytest.fixture
def loaded_test_documents(iris_sql_connection):
    """Load test documents into RAG.SourceDocuments for integration testing.

    This fixture ensures MCP and pipeline integration tests have data to query.
    """
    import sys

    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager
    from iris_vector_rag.core.models import Document
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    print("\n🔧 [FIXTURE] loaded_test_documents starting...", file=sys.stderr)

    # Create documents
    documents = [
        Document(
            page_content="Diabetes is a chronic disease that affects how your body processes blood sugar. Common symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision.",
            metadata={"source": "test_medical.pdf", "page": 1, "topic": "diabetes"},
        ),
        Document(
            page_content="Type 2 diabetes symptoms often develop slowly over several years. Many people don't notice symptoms at first. Early signs include increased hunger, dry mouth, and slow-healing sores.",
            metadata={"source": "test_medical.pdf", "page": 2, "topic": "diabetes"},
        ),
        Document(
            page_content="IRIS database provides native vector search capabilities with HNSW indexing. This enables high-performance semantic search for RAG applications.",
            metadata={"source": "test_technical.pdf", "page": 1, "topic": "database"},
        ),
        Document(
            page_content="RAG pipelines combine retrieval and generation. The retrieval step uses vector similarity search to find relevant documents.",
            metadata={"source": "test_technical.pdf", "page": 2, "topic": "rag"},
        ),
        Document(
            page_content="Knowledge graphs represent entities and relationships. Entity extraction identifies important concepts from text documents.",
            metadata={
                "source": "test_graph.pdf",
                "page": 1,
                "topic": "knowledge_graph",
            },
        ),
    ]

    try:
        # CRITICAL: Drop and recreate table with correct VECTOR schema
        from sqlalchemy import text

        from iris_vector_rag.storage.schema_manager import SchemaManager

        # Create managers
        conn_manager = ConnectionManager()
        config_manager = ConfigurationManager()
        schema_mgr = SchemaManager(conn_manager, config_manager)

        # Check if migration is needed
        print("🔍 [FIXTURE] Checking if schema migration needed...", file=sys.stderr)
        needs_migration = schema_mgr.needs_migration(
            "SourceDocuments", pipeline_type="basic"
        )
        print(f"   Migration needed: {needs_migration}", file=sys.stderr)

        if needs_migration:
            print(
                "🔄 [FIXTURE] Migrating table schema to VECTOR datatype...",
                file=sys.stderr,
            )
            # The fixture deletes this table's rows on the next lines anyway.
            schema_mgr.migrate_table(
                "SourceDocuments",
                preserve_data=False,
                pipeline_type="basic",
                allow_destructive=True,
            )
            print("✅ [FIXTURE] Schema migrated", file=sys.stderr)
        else:
            print(
                "🔧 [FIXTURE] Ensuring schema with VECTOR datatype...", file=sys.stderr
            )
            schema_mgr.ensure_table_schema("SourceDocuments", pipeline_type="basic")
            print("✅ [FIXTURE] Schema ensured", file=sys.stderr)

        # Clean existing data
        print("🧹 [FIXTURE] Cleaning existing data...", file=sys.stderr)
        try:
            iris_sql_connection.execute(text("DELETE FROM RAG.SourceDocuments"))
            iris_sql_connection.commit()
            print("✅ [FIXTURE] Cleaned", file=sys.stderr)
        except Exception as clean_err:
            print(f"⚠️  [FIXTURE] Clean failed: {clean_err}", file=sys.stderr)

        pipeline = BasicRAGPipeline(conn_manager, config_manager)

        # Load documents into database
        print(f"📥 [FIXTURE] Loading {len(documents)} documents...", file=sys.stderr)
        pipeline.load_documents(documents=documents)
        print(f"✅ [FIXTURE] Loaded {len(documents)} documents", file=sys.stderr)

        yield documents

    finally:
        # Cleanup is handled by cleanup_test_data autouse fixture
        pass


@pytest.fixture(autouse=True)
def cleanup_test_data(iris_sql_connection):
    """Cleanup test data after each integration test."""
    yield

    # Cleanup test tables
    from sqlalchemy import text

    test_tables = [
        "integration_test_vectors",
        "test_documents",
        "test_entities",
        "test_relationships",
    ]

    for table in test_tables:
        try:
            iris_sql_connection.execute(text(f"DROP TABLE IF EXISTS {table}"))
            iris_sql_connection.commit()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
def iris_schema_manager(iris_sql_connection, iris_connection_config):
    """Schema manager for integration tests."""
    from iris_vector_rag.storage.schema_manager import SchemaManager

    settings = {
        "database.iris.host": iris_connection_config["host"],
        "database.iris.port": iris_connection_config["port"],
        "database.iris.username": iris_connection_config["username"],
        "database.iris.password": iris_connection_config["password"],
        "database.iris.namespace": iris_connection_config["namespace"],
    }
    mock_config = Mock()
    mock_config.get = Mock(
        side_effect=lambda key, default=None: settings.get(key, default)
    )

    manager = SchemaManager(config=mock_config)
    manager.connection = iris_sql_connection  # Use test connection
    yield manager
