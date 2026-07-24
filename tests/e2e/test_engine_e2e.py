"""
E2E Tests for IRISVectorEngine (feature 080-ivr-engine).

Tests the new unified engine object against a real IRIS instance.
All fixtures are scope="function" per AGENTS.md KNOWN PAIN POINTS.
"""

import pytest

from iris_vector_rag import (
    IRISVectorEngine,
    create_pipeline,
    create_validated_pipeline,
)
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore


@pytest.fixture
def engine():
    """Fresh IRISVectorEngine per test (scope=function — avoids DBAPI C-ext crash)."""
    return IRISVectorEngine.from_config()


@pytest.mark.e2e
@pytest.mark.requires_database
def test_from_config_connection_is_live(engine):
    """US1 SC-001: from_config() opens a real IRIS connection."""
    conn = engine.connection
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    row = cursor.fetchone()
    assert row is not None
    value = row[0]
    cursor.close()
    assert value == 1


@pytest.mark.e2e
@pytest.mark.requires_database
def test_vector_store_schema_prefix_matches_engine(engine):
    """US4 SC-005: engine.vector_store.schema_manager.schema_prefix == engine.schema_prefix."""
    vs = engine.vector_store
    assert isinstance(vs, IRISVectorStore)
    assert vs.schema_manager.schema_prefix == engine.schema_prefix


@pytest.mark.e2e
@pytest.mark.requires_database
def test_basic_pipeline_from_engine_query_shape():
    """US1 SC-002: BasicRAGPipeline(engine) query returns standard response shape."""
    engine = IRISVectorEngine.from_config()
    pipeline = BasicRAGPipeline(engine)
    result = pipeline.query("What is diabetes?", top_k=3)
    assert isinstance(result, dict)
    for key in ("answer", "retrieved_documents", "contexts", "sources", "metadata"):
        assert key in result, f"Missing key in query response: {key}"


@pytest.mark.e2e
@pytest.mark.requires_database
def test_create_pipeline_with_engine_kwarg():
    """US2: create_pipeline('basic', engine=engine) works end-to-end."""
    engine = IRISVectorEngine.from_config()
    pipeline = create_pipeline("basic", engine=engine)
    assert isinstance(pipeline, BasicRAGPipeline)
    result = pipeline.query("What causes hypertension?", top_k=3)
    assert "answer" in result
    assert "retrieved_documents" in result


@pytest.mark.e2e
@pytest.mark.requires_database
def test_create_validated_pipeline_with_engine_kwarg():
    """US2: create_validated_pipeline(engine=engine) constructs without error."""
    engine = IRISVectorEngine.from_config()
    pipeline = create_validated_pipeline(
        pipeline_type="basic",
        engine=engine,
        validate_requirements=True,
        auto_setup=True,
    )
    assert isinstance(pipeline, BasicRAGPipeline)


@pytest.mark.e2e
@pytest.mark.requires_database
def test_raw_connection_engine_wraps_correctly():
    """US3: IRISVectorEngine(raw_conn) wraps connection in ExternalConnectionWrapper."""
    from iris_vector_rag import ExternalConnectionWrapper

    engine = IRISVectorEngine.from_config()
    raw_conn = engine.connection

    engine2 = IRISVectorEngine(raw_conn, schema_prefix="RAG")
    assert isinstance(engine2.connection_manager, ExternalConnectionWrapper)

    vs = engine2.vector_store
    assert isinstance(vs, IRISVectorStore)


@pytest.mark.e2e
@pytest.mark.requires_database
def test_schema_prefix_flows_through_engine():
    """US4 SC-001/SC-003: schema_prefix set on engine propagates to vector_store."""
    engine = IRISVectorEngine.from_config(schema_prefix="RAG")
    assert engine.schema_prefix == "RAG"
    assert engine.vector_store.schema_manager.schema_prefix == "RAG"


@pytest.mark.e2e
@pytest.mark.requires_database
def test_engine_lazy_no_connection_before_access():
    """FR-008: engine does not open DB connection at from_config() time."""
    engine = IRISVectorEngine.from_config()
    assert engine._connection is None
    assert engine._vector_store is None
    _ = engine.connection
    assert engine._connection is not None
