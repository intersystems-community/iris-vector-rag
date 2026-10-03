"""iris-vector-graph 4.x: graph-scoped storage, and what IVR does about an older install.

ivg 4.0 re-keyed `kg_NodeEmbeddings` on `(graph_id, node_id)` and dropped its `id`
column. A 2.x install keeps working at the SQL level -- `WHERE id = ?` is simply gone
from ivg, and its `WHERE node_id = ?` fails on the old table -- so the old layout has to
be *reported*, by name, with the call that fixes it. IVR never runs the data migration on
its own: `upgrade_to_4_0_0` moves and quarantines rows, which is an operator's decision.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.integrations import ivg
from iris_vector_rag.integrations.ivg import (
    IVG_4X_REQUIRED_COLUMNS,
    IVG_REQUIRED_TABLES,
    IVG_UPGRADE_HINT,
    MIN_IVG_VERSION,
    assert_ivg_compatible,
    pre_4_layout_tables,
)
from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import HybridRetrievalMethods
from iris_vector_rag.storage.schema_manager import SchemaManager

LEGACY = {
    "nodes": {"node_id", "created_at", "graph_id"},
    "rdf_labels": {"s", "label"},
    "rdf_props": {"s", "key", "val"},
    "rdf_edges": {"edge_id", "s", "p", "o_id", "qualifiers", "graph_id"},
    "kg_NodeEmbeddings": {"id", "emb", "metadata"},
    "kg_NodeEmbeddings_optimized": {"id", "emb", "metadata"},
}

CURRENT = {
    "nodes": {"node_id", "created_at", "graph_id"},
    "rdf_labels": {"graph_id", "s", "label"},
    "rdf_props": {"graph_id", "s", "key", "val"},
    "rdf_edges": {"edge_id", "s", "p", "o_id", "qualifiers", "graph_id", "ekey"},
    "kg_NodeEmbeddings": {"emb_rowid", "graph_id", "node_id", "emb", "metadata"},
    "kg_NodeEmbeddings_optimized": {
        "emb_rowid",
        "graph_id",
        "node_id",
        "emb",
        "metadata",
    },
}


# ---------------------------------------------------------------------------
# The version floor and the layout rule
# ---------------------------------------------------------------------------


def test_minimum_version_is_the_layout_break():
    """4.0.0 is where the storage layout changed; nothing older can read it."""
    assert MIN_IVG_VERSION == "4.0.0"


def test_a_2x_module_is_refused_with_the_floor_in_the_message():
    module = types.SimpleNamespace(__version__="2.18.5")

    with pytest.raises(ImportError, match=r">=4\.0\.0; found 2\.18\.5"):
        assert_ivg_compatible(module)


def test_a_4x_module_is_accepted():
    module = types.SimpleNamespace(__version__="4.1.1")

    assert assert_ivg_compatible(module) == "4.1.1"


_SURFACE = (
    "IRISGraphEngine",
    "GraphStore",
    "IVGResult",
    "EmbedSelector",
    "IndexConfig",
    "EngineStatus",
)


def _unversioned(name):
    """A checkout with no version to read: only its surface can vouch for it."""
    module = types.ModuleType(name)
    for attr in _SURFACE:
        setattr(module, attr, object())
    return module


def test_an_unversioned_2x_surface_is_refused():
    """2.x had the same top-level names; only 4.x ships the layout migration."""
    module = _unversioned("fake_ivg_2x")

    with (
        patch.object(ivg, "get_ivg_version", return_value=None),
        pytest.raises(ImportError, match="found unknown"),
    ):
        assert_ivg_compatible(module)


def test_an_unversioned_4x_surface_is_accepted():
    module = _unversioned("fake_ivg_4x")
    migrations = types.ModuleType("fake_ivg_4x.migrations")
    migrations.upgrade_to_4_0_0 = lambda conn, **kw: None

    with (
        patch.object(ivg, "get_ivg_version", return_value=None),
        patch.dict("sys.modules", {"fake_ivg_4x.migrations": migrations}),
    ):
        assert assert_ivg_compatible(module) is None


def test_every_required_table_has_a_layout_rule():
    """A table IVR requires but cannot judge would pass the check by omission."""
    assert set(IVG_4X_REQUIRED_COLUMNS) == set(IVG_REQUIRED_TABLES)


def test_the_2x_layout_is_reported_table_by_table():
    """`nodes` and `rdf_edges` already had graph_id in late 2.x; the other four did not."""
    assert pre_4_layout_tables(LEGACY) == [
        "rdf_labels",
        "rdf_props",
        "kg_NodeEmbeddings",
        "kg_NodeEmbeddings_optimized",
    ]


def test_the_4x_layout_reports_nothing():
    assert pre_4_layout_tables(CURRENT) == []


def test_a_missing_table_is_not_a_layout_finding():
    """Absent tables are `missing_tables`' business; counting them twice misleads."""
    columns = dict(CURRENT)
    del columns["kg_NodeEmbeddings_optimized"]
    columns["rdf_props"] = set()

    assert pre_4_layout_tables(columns) == []


def test_column_names_are_compared_case_insensitively():
    """The catalog reports names in declared case; `NODE_ID` is still node_id."""
    columns = dict(CURRENT)
    columns["kg_NodeEmbeddings"] = {"EMB_ROWID", "Graph_Id", "NODE_ID", "EMB"}

    assert pre_4_layout_tables(columns) == []


def test_the_upgrade_hint_names_both_calls():
    assert "upgrade_iris_vector_graph_storage" in IVG_UPGRADE_HINT
    assert "upgrade_to_4_0_0" in IVG_UPGRADE_HINT


# ---------------------------------------------------------------------------
# SchemaManager: reporting and the explicit upgrade
# ---------------------------------------------------------------------------


def _manager(columns_by_table):
    manager = SchemaManager.__new__(SchemaManager)
    manager._detect_iris_vector_graph = lambda: True
    manager._ivg_table_exists = lambda table: bool(columns_by_table.get(table))
    manager._ivg_columns = lambda table: set(columns_by_table.get(table, set()))
    return manager


@pytest.fixture
def compatible():
    with patch(
        "iris_vector_rag.storage.schema_manager.assert_ivg_compatible",
        return_value="4.1.1",
    ):
        yield


def test_legacy_layout_tables_reads_graph_kg(compatible):
    manager = _manager(LEGACY)

    assert manager._ivg_legacy_layout_tables() == pre_4_layout_tables(LEGACY)


def test_ivg_columns_reads_the_graph_kg_snapshot():
    """The column read goes through `_snapshot`, in Graph_KG, not the RAG schema."""
    manager = SchemaManager.__new__(SchemaManager)
    snapshot = types.SimpleNamespace(
        columns=[{"COLUMN_NAME": "Node_Id"}, {"COLUMN_NAME": "graph_id"}]
    )
    calls = []

    def fake_snapshot(table, *, schema=None):
        calls.append((table, schema))
        return snapshot if table == "kg_NodeEmbeddings" else None

    manager._snapshot = fake_snapshot

    assert manager._ivg_columns("kg_NodeEmbeddings") == {"node_id", "graph_id"}
    assert manager._ivg_columns("rdf_props") == set()
    assert calls == [
        ("kg_NodeEmbeddings", ivg.IVG_SCHEMA),
        ("rdf_props", ivg.IVG_SCHEMA),
    ]


def test_validation_fails_on_the_2x_layout_and_names_the_fix(compatible):
    result = _manager(LEGACY).validate_graph_prerequisites("graphrag")

    assert result.is_valid is False
    assert result.missing_tables == []
    assert result.legacy_layout_tables == pre_4_layout_tables(LEGACY)
    assert "upgrade_iris_vector_graph_storage" in result.error_message
    assert "kg_NodeEmbeddings" in result.error_message


def test_validation_passes_on_the_4x_layout(compatible):
    result = _manager(CURRENT).validate_graph_prerequisites("graphrag")

    assert result.is_valid is True
    assert result.legacy_layout_tables == []


def test_initialization_reports_the_2x_layout_without_migrating(compatible):
    """`initialize_schema` runs; the row-moving migration does not."""
    manager = _manager(LEGACY)
    manager._initialize_ivg_schema = lambda: {"tables_created": True}

    with patch("iris_vector_graph.migrations.upgrade_to_4_0_0") as upgrade:
        result = manager.ensure_iris_vector_graph_tables("graphrag")

    upgrade.assert_not_called()
    assert result.legacy_layout_tables == pre_4_layout_tables(LEGACY)
    assert "upgrade_iris_vector_graph_storage" in result.error_messages[ivg.IVG_SCHEMA]
    # The tables do exist; the finding is about their shape, not their presence.
    assert all(result.tables_created.values())


def test_initialization_on_the_4x_layout_has_no_layout_error(compatible):
    manager = _manager(CURRENT)
    manager._initialize_ivg_schema = lambda: {"tables_created": True}

    result = manager.ensure_iris_vector_graph_tables("graphrag")

    assert result.legacy_layout_tables == []
    assert result.error_messages == {}


def test_a_failed_layout_read_is_reported_not_swallowed(compatible):
    manager = _manager(CURRENT)
    manager._initialize_ivg_schema = lambda: {"tables_created": True}

    def broken():
        raise RuntimeError("catalog unavailable")

    manager._ivg_legacy_layout_tables = broken

    result = manager.ensure_iris_vector_graph_tables("graphrag")

    assert "catalog unavailable" in result.error_messages[ivg.IVG_SCHEMA]


def _upgrade_manager():
    manager = SchemaManager.__new__(SchemaManager)
    manager.base_embedding_dimension = 384
    connection = object()
    manager.connection_manager = MagicMock()
    manager.connection_manager.get_connection.return_value = connection
    return manager, connection


def test_upgrade_initializes_schema_then_runs_the_ivg_migration():
    """Order is ivg's own: `initialize_schema` first, the data migration second."""
    manager, connection = _upgrade_manager()
    order = []
    engine = MagicMock()
    engine.initialize_schema.side_effect = lambda: order.append("init") or {}
    report = object()

    def fake_upgrade(conn, **kwargs):
        order.append(("upgrade", conn, kwargs))
        return report

    with (
        patch(
            "iris_vector_rag.storage.schema_manager.create_graph_engine",
            return_value=engine,
        ) as factory,
        patch(
            "iris_vector_graph.migrations.upgrade_to_4_0_0", side_effect=fake_upgrade
        ),
    ):
        result = manager.upgrade_iris_vector_graph_storage()

    assert result is report
    factory.assert_called_once_with(connection, embedding_dimension=384)
    assert order == ["init", ("upgrade", connection, {"dry_run": False})]


def test_upgrade_dry_run_is_ivgs_own_and_writes_no_ddl():
    """A dry run writes nothing, and `initialize_schema` writes DDL, so it is not run.

    ivg 4.1.1 answers a step it cannot predict on a pre-4.0 install with
    `blocked_because` on that step, so one call previews the whole sequence.
    """
    manager, connection = _upgrade_manager()
    report = object()

    with (
        patch("iris_vector_rag.storage.schema_manager.create_graph_engine") as factory,
        patch(
            "iris_vector_graph.migrations.upgrade_to_4_0_0", return_value=report
        ) as upgrade,
    ):
        result = manager.upgrade_iris_vector_graph_storage(dry_run=True)

    assert result is report
    factory.assert_not_called()
    upgrade.assert_called_once_with(connection, dry_run=True)


def test_a_blocked_step_reaches_the_caller_unchanged():
    """The reason a step cannot be previewed is ivg's to state; IVR passes it on."""
    from iris_vector_graph.migrations import StepResult, UpgradeReport

    manager, _ = _upgrade_manager()
    blocked = UpgradeReport(
        steps=(
            StepResult("embeddings", report="ok"),
            StepResult("docs", blocked_because="nodes has no graph_id column"),
        )
    )

    with patch("iris_vector_graph.migrations.upgrade_to_4_0_0", return_value=blocked):
        report = manager.upgrade_iris_vector_graph_storage(dry_run=True)

    assert report.blocked == ("docs",)
    assert "no graph_id column" in report["docs"].blocked_because


def test_a_real_run_does_not_swallow_a_step_failure():
    manager, _ = _upgrade_manager()

    with (
        patch("iris_vector_rag.storage.schema_manager.create_graph_engine"),
        patch(
            "iris_vector_graph.migrations.upgrade_to_4_0_0",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        manager.upgrade_iris_vector_graph_storage()


def test_upgrade_forwards_migration_options():
    manager, connection = _upgrade_manager()
    resolver = lambda item: None  # noqa: E731

    with (
        patch("iris_vector_rag.storage.schema_manager.create_graph_engine"),
        patch("iris_vector_graph.migrations.upgrade_to_4_0_0") as upgrade,
    ):
        manager.upgrade_iris_vector_graph_storage(
            steps=("embeddings",), vector_resolver=resolver
        )

    upgrade.assert_called_once_with(
        connection, dry_run=False, steps=("embeddings",), vector_resolver=resolver
    )


# ---------------------------------------------------------------------------
# Retrieval: graph-scoped search and the embedding route (ivg 4.1 keywords)
# ---------------------------------------------------------------------------


def _retrieval():
    engine = MagicMock()
    engine.kg_RRF_FUSE.return_value = []
    engine.kg_TXT.return_value = []
    engine.kg_KNN_VEC.return_value = []
    methods = HybridRetrievalMethods(
        iris_engine=engine,
        fusion_engine=None,
        text_engine=None,
        vector_optimizer=None,
        embedding_manager=None,
    )
    return methods, engine


def _noop_content(_doc_id):
    return ""


def test_default_calls_pass_no_graph_keywords():
    """No `graph` asked for, none sent: the default graph, as before 4.x."""
    methods, engine = _retrieval()

    methods.retrieve_via_rrf("q", 3, _noop_content, vector_query="[1.0]")
    methods.retrieve_via_enhanced_text("q", 3, _noop_content)
    methods.retrieve_via_hnsw_vector("q", 3, _noop_content, vector_query="[1.0]")

    assert "graph" not in engine.kg_RRF_FUSE.call_args.kwargs
    assert "graph" not in engine.kg_TXT.call_args.kwargs
    assert "graph" not in engine.kg_KNN_VEC.call_args.kwargs
    assert "model_key" not in engine.kg_KNN_VEC.call_args.kwargs


def test_graph_scopes_all_three_searches():
    methods, engine = _retrieval()

    methods.retrieve_via_rrf(
        "q", 3, _noop_content, vector_query="[1.0]", graph="tenant-a"
    )
    methods.retrieve_via_enhanced_text("q", 3, _noop_content, graph="tenant-a")
    methods.retrieve_via_hnsw_vector(
        "q", 3, _noop_content, vector_query="[1.0]", graph="tenant-a"
    )

    assert engine.kg_RRF_FUSE.call_args.kwargs["graph"] == "tenant-a"
    assert engine.kg_TXT.call_args.kwargs["graph"] == "tenant-a"
    assert engine.kg_KNN_VEC.call_args.kwargs["graph"] == "tenant-a"


def test_model_key_selects_the_embedding_route():
    methods, engine = _retrieval()

    methods.retrieve_via_hnsw_vector(
        "q", 3, _noop_content, vector_query="[1.0]", model_key="minilm"
    )

    assert engine.kg_KNN_VEC.call_args.kwargs["model_key"] == "minilm"
