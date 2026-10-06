"""E2E: upgrading Graph_KG to the iris-vector-graph 4.x layout, then using it.

Runs against the IRIS the suite provides (`iris_target`). The first run on an
install that predates ivg 4.0 exercises the whole path: the old layout is reported by
name, the dry run previews without writing, the real run migrates, and validation comes
back clean. Every later run finds the 4.x layout already there and checks that the
upgrade is a re-runnable no-op. Either way the test ends with a graph-scoped
write / search / delete roundtrip through IVR's retrieval layer.
"""

from __future__ import annotations

import uuid

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.integrations.ivg import create_graph_engine
from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import HybridRetrievalMethods
from iris_vector_rag.storage.schema_manager import SchemaManager

pytestmark = pytest.mark.integration

DIM = 384


@pytest.fixture(scope="module")
def schema_manager():
    migrations = pytest.importorskip("iris_vector_graph.migrations")
    assert hasattr(migrations, "upgrade_to_4_0_0")
    # `iris_target` (autouse for tests/integration) exports IRIS_* for the session,
    # which ConfigurationManager resolves into the connection identity.
    config = ConfigurationManager()
    return SchemaManager(ConnectionManager(config), config)


@pytest.fixture(scope="module")
def upgraded(schema_manager):
    """Report, preview, migrate -- in that order, once per module."""
    from iris_vector_graph.migrations import UPGRADE_STEPS

    before = schema_manager._ivg_legacy_layout_tables()
    init = schema_manager.ensure_iris_vector_graph_tables("graphrag")
    assert init.legacy_layout_tables == before

    preview = schema_manager.upgrade_iris_vector_graph_storage(dry_run=True)
    # The dry run wrote nothing: the layout is exactly as it was.
    assert schema_manager._ivg_legacy_layout_tables() == before

    report = schema_manager.upgrade_iris_vector_graph_storage()
    return {
        "before": before,
        "init": init,
        "preview": preview,
        "report": report,
        "steps": tuple(UPGRADE_STEPS),
    }


def test_the_old_layout_was_reported_with_the_call_that_fixes_it(upgraded):
    init = upgraded["init"]
    if not upgraded["before"]:
        assert "Pre-4.0" not in init.error_messages.get("Graph_KG", "")
        return
    assert "upgrade_iris_vector_graph_storage" in init.error_messages["Graph_KG"]


def test_the_dry_run_previews_every_step(upgraded):
    """Every step is accounted for: previewed, skipped as done, or blocked with a reason."""
    preview = upgraded["preview"]

    assert preview.names == upgraded["steps"]
    for step in preview.steps:
        assert (
            step.report is not None or step.skipped_because or step.blocked_because
        ), step


def test_the_real_run_leaves_no_old_layout(upgraded, schema_manager):
    assert upgraded["report"].names == upgraded["steps"]
    assert schema_manager._ivg_legacy_layout_tables() == []

    validation = schema_manager.validate_graph_prerequisites("graphrag")
    assert validation.is_valid, validation.error_message
    assert validation.legacy_layout_tables == []


def test_upgrading_again_changes_nothing(upgraded, schema_manager):
    again = schema_manager.upgrade_iris_vector_graph_storage()

    assert again.names == upgraded["steps"]
    assert schema_manager._ivg_legacy_layout_tables() == []


def _unit(index: int):
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


def _retrieval(schema_manager):
    engine = create_graph_engine(
        schema_manager.connection_manager.get_connection(), embedding_dimension=DIM
    )
    methods = HybridRetrievalMethods(
        iris_engine=engine,
        fusion_engine=None,
        text_engine=None,
        vector_optimizer=None,
        embedding_manager=None,
    )

    def found(vector, **scope):
        docs, method = methods.retrieve_via_hnsw_vector(
            "unused",
            5,
            lambda doc_id: doc_id,
            vector_query=str(vector),
            **scope,
        )
        assert method == "hnsw_vector"
        return {doc.metadata["entity_id"] for doc in docs}

    return engine, found


def _node_id():
    return f"ivr-e2e-{uuid.uuid4().hex[:10]}"


def test_graph_scoped_search_roundtrip(upgraded, schema_manager):
    """The same node ID in two graphs; `graph=` picks which one a search sees."""
    engine, found = _retrieval(schema_manager)
    node_id = _node_id()
    graphs = {f"{node_id}-a": _unit(0), f"{node_id}-b": _unit(1)}

    try:
        for graph, vector in graphs.items():
            assert engine.create_node(node_id, labels=["IVRE2E"], graph=graph)
            assert engine.store_embedding(node_id, vector, graph=graph)

        # Each graph sees its own copy; neither sees the other's vector.
        for graph, vector in graphs.items():
            assert node_id in found(vector, graph=graph)
        assert node_id not in found(_unit(0), graph="ivr-e2e-no-such-graph")
    finally:
        # `graph=` reaches the routed `kg_emb_<hash>` table a named graph's vectors
        # live in; ivg 4.1.0's `delete_node` did not, and returned False.
        for graph in graphs:
            assert engine.delete_node(node_id, graph=graph)

    for graph, vector in graphs.items():
        assert node_id not in found(vector, graph=graph)


def test_default_graph_delete_node_roundtrip(upgraded, schema_manager):
    """The path IVR's pipeline takes: default graph, `delete_node` to tear down."""
    engine, found = _retrieval(schema_manager)
    node_id = _node_id()

    try:
        assert engine.create_node(node_id, labels=["IVRE2E"])
        assert engine.store_embedding(node_id, _unit(2))
        assert node_id in found(_unit(2))
    finally:
        assert engine.delete_node(node_id)

    assert node_id not in found(_unit(2))
