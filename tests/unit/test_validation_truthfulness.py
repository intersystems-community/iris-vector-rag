"""Validation and auto-setup report what is true (VM-02, 07, 08, 09, 11, 12, 14, 15).

Each test names a way the validator or the setup orchestrator used to say something
that was not so: a required flag it never read, a count that failed and was reported
as a missing table, a fake row inserted so a minimum would be met, a verification
that could not run and was reported as success.

No database and no embedding model: the cursor, the catalog and the embedder are
recordings. The same behaviours against a real IRIS are in
`tests/e2e/test_validation_truthfulness_e2e.py`.
"""

from __future__ import annotations

import ast
import inspect
import re
from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_vector_rag.storage.catalog import CatalogSnapshot, CatalogUnavailable
from iris_vector_rag.validation import factory as factory_module
from iris_vector_rag.validation import orchestrator as orchestrator_module
from iris_vector_rag.validation.orchestrator import SetupError, SetupOrchestrator
from iris_vector_rag.validation.requirements import (
    PIPELINE_REQUIREMENTS_REGISTRY,
    EmbeddingRequirement,
    PipelineRequirements,
    TableRequirement,
    get_pipeline_requirements,
)
from iris_vector_rag.validation.validator import PreConditionValidator


class ScriptedCursor:
    """A cursor whose `COUNT(*)` answers, or fails, per table, from a script."""

    def __init__(self, counts=None, failing=()):
        self.counts = dict(counts or {})
        self.failing = set(failing)
        self.statements: list[str] = []
        self._last = None

    def execute(self, sql, params=None):
        sql = " ".join(str(sql).split())
        self.statements.append(sql)
        if "COUNT(" in sql:
            for table in self.failing:
                if table in sql:
                    raise RuntimeError("connection reset by peer")
            for table, count in self.counts.items():
                if table in sql:
                    self._last = (count,)
                    return
            raise RuntimeError(f"<TABLE DOES NOT EXIST> {sql}")
        self._last = None

    def fetchone(self):
        return self._last

    def fetchall(self):
        return []

    def close(self):
        pass


class Connections:
    def __init__(self, cursor):
        self.cursor_ = cursor

    def get_connection(self):
        connection = Mock()
        connection.cursor.return_value = self.cursor_
        return connection


class CatalogDouble:
    """Answers `snapshot(schema, table)`: present, absent, or unreadable."""

    def __init__(self, present=(), unreadable=False):
        self.present = {t.lower() for t in present}
        self.unreadable = unreadable
        self.asked: list[tuple[str, str]] = []

    def snapshot(self, schema, table):
        self.asked.append((schema, table))
        if self.unreadable:
            raise CatalogUnavailable("catalog query failed: closed", schema, table)
        if table.lower() in self.present:
            return CatalogSnapshot(schema=schema, table=table)
        return None


class Reqs(PipelineRequirements):
    """SourceDocuments required, Communities listed as required but flagged not."""

    def __init__(self, tables, embeddings=()):
        self._tables = tables
        self._embeddings = list(embeddings)

    @property
    def pipeline_name(self):
        return "unit"

    @property
    def required_tables(self):
        return self._tables

    @property
    def required_embeddings(self):
        return self._embeddings


def validator_for(cursor, catalog):
    return PreConditionValidator(Connections(cursor), catalog=catalog)


# --- VM-08: `.required` is honoured --------------------------------------------


def test_a_table_flagged_not_required_does_not_fail_the_pipeline():
    requirements = Reqs(
        [
            TableRequirement("SourceDocuments", "RAG", "docs", min_rows=1),
            TableRequirement("Communities", "RAG", "optional", required=False),
        ]
    )
    cursor = ScriptedCursor(counts={"RAG.SourceDocuments": 3})
    report = validator_for(
        cursor, CatalogDouble(present=["SourceDocuments"])
    ).validate_pipeline_requirements(requirements)

    assert report.overall_valid, report.summary
    assert not report.table_validations["Communities_optional"].is_valid
    assert "Communities" not in report.summary or "optional" in report.summary.lower()
    assert "Run setup orchestrator" not in " ".join(report.setup_suggestions)


def test_a_required_table_missing_still_fails_and_is_named():
    requirements = Reqs([TableRequirement("SourceDocuments", "RAG", "docs")])
    report = validator_for(
        ScriptedCursor(), CatalogDouble()
    ).validate_pipeline_requirements(requirements)
    assert not report.overall_valid
    assert "SourceDocuments" in report.summary


def test_an_embedding_flagged_not_required_does_not_fail_the_pipeline():
    requirements = Reqs(
        [],
        embeddings=[
            EmbeddingRequirement(
                "soft", "RAG.SourceDocuments", "embedding", "x", required=False
            )
        ],
    )
    report = validator_for(
        ScriptedCursor(), CatalogDouble()
    ).validate_pipeline_requirements(requirements)
    assert report.overall_valid, report.summary


def test_graphrag_is_valid_without_communities():
    requirements = get_pipeline_requirements("graphrag")
    communities = [t for t in requirements.required_tables if t.name == "Communities"]
    assert communities and communities[0].required is False

    counts = {
        "RAG.SourceDocuments": 2,
        "RAG.Entities": 2,
        "RAG.EntityRelationships": 1,
    }
    cursor = ScriptedCursor(counts=counts)
    catalog = CatalogDouble(
        present=["SourceDocuments", "Entities", "EntityRelationships"]
    )
    validator = validator_for(cursor, catalog)
    with patch.object(
        PreConditionValidator,
        "_validate_embedding_requirement",
        return_value=Mock(is_valid=True, suggestions=[]),
    ):
        report = validator.validate_pipeline_requirements(requirements)
    assert report.overall_valid, report.summary


# --- VM-11: a failed count is not a missing table -------------------------------


def test_a_failed_count_on_an_existing_table_is_a_count_failure():
    requirements = Reqs([TableRequirement("SourceDocuments", "RAG", "docs")])
    cursor = ScriptedCursor(failing=["RAG.SourceDocuments"])
    report = validator_for(
        cursor, CatalogDouble(present=["SourceDocuments"])
    ).validate_pipeline_requirements(requirements)

    result = report.table_validations["SourceDocuments"]
    assert not result.is_valid
    assert result.details["error_kind"] == "count_failed"
    assert "does not exist" not in result.message
    assert "connection reset" in result.message
    assert not any(s.startswith("Create table") for s in result.suggestions)


def test_a_count_that_fails_on_an_absent_table_is_a_missing_table():
    requirements = Reqs([TableRequirement("SourceDocuments", "RAG", "docs")])
    report = validator_for(
        ScriptedCursor(), CatalogDouble()
    ).validate_pipeline_requirements(requirements)
    result = report.table_validations["SourceDocuments"]
    assert result.details["error_kind"] == "missing_table"
    assert "does not exist" in result.message
    assert "Create table RAG.SourceDocuments" in result.suggestions


def test_when_the_catalog_cannot_be_read_the_table_is_not_called_missing():
    requirements = Reqs([TableRequirement("SourceDocuments", "RAG", "docs")])
    cursor = ScriptedCursor(failing=["RAG.SourceDocuments"])
    report = validator_for(
        cursor, CatalogDouble(unreadable=True)
    ).validate_pipeline_requirements(requirements)
    result = report.table_validations["SourceDocuments"]
    assert result.details["error_kind"] == "count_failed"
    assert "does not exist" not in result.message


def test_the_catalog_is_asked_about_the_requirements_own_schema():
    requirements = Reqs([TableRequirement("rdf_labels", "SQLUSER", "ivg")])
    catalog = CatalogDouble()
    validator_for(ScriptedCursor(), catalog).validate_pipeline_requirements(
        requirements
    )
    assert catalog.asked == [("SQLUSER", "rdf_labels")]


# --- VM-07: requirements point at columns that exist ---------------------------


def test_every_embedding_requirement_names_a_canonical_column():
    from iris_vector_rag.storage.table_shape import (
        CANONICAL_CHUNK_SHAPE,
        CANONICAL_DOCUMENT_SHAPE,
    )

    canonical = {
        "RAG.SourceDocuments": set(CANONICAL_DOCUMENT_SHAPE.columns),
        "RAG.DocumentChunks": set(CANONICAL_CHUNK_SHAPE.columns),
    }
    for name in PIPELINE_REQUIREMENTS_REGISTRY:
        requirements = get_pipeline_requirements(name)
        for embedding in (
            requirements.required_embeddings + requirements.optional_embeddings
        ):
            if embedding.table in canonical:
                assert embedding.column in canonical[embedding.table], (
                    name,
                    embedding,
                )
            # The standard Entities table has no vector column, so asking for one
            # was a requirement that could never be met.
            assert embedding.table != "RAG.Entities", (name, embedding)


def test_the_embedding_suggestion_names_a_method_that_exists():
    requirements = Reqs(
        [],
        embeddings=[EmbeddingRequirement("e", "RAG.SourceDocuments", "embedding", "x")],
    )
    cursor = ScriptedCursor(counts={"RAG.SourceDocuments": 0})
    report = validator_for(cursor, CatalogDouble()).validate_pipeline_requirements(
        requirements
    )
    named = [s for s in report.setup_suggestions if "SetupOrchestrator." in s]
    assert named, report.setup_suggestions
    for suggestion in named:
        method = suggestion.split("SetupOrchestrator.")[1].split("(")[0]
        assert hasattr(SetupOrchestrator, method), suggestion


# --- orchestrator fixtures -------------------------------------------------------


class Recording:
    def __init__(self, ones=(), alls=(), fail_on=()):
        self._ones = list(ones)
        self._alls = list(alls)
        self.fail_on = tuple(fail_on)
        self.statements: list[str] = []

    def execute(self, sql, params=None):
        sql = " ".join(str(sql).split())
        self.statements.append(sql)
        if any(fragment in sql for fragment in self.fail_on):
            raise RuntimeError("boom")

    def fetchone(self):
        return self._ones.pop(0) if self._ones else None

    def fetchall(self):
        return self._alls.pop(0) if self._alls else []

    def close(self):
        pass


def orchestrator_over(cursor, llm_func=None):
    connection = Mock()
    connection.cursor.return_value = cursor
    connections = Mock()
    connections.get_connection.return_value = connection
    config = Mock()
    config.get.return_value = {}
    with (
        patch("iris_vector_rag.validation.orchestrator.EmbeddingManager"),
        patch("iris_vector_rag.validation.orchestrator.PreConditionValidator"),
    ):
        return SetupOrchestrator(connections, config, llm_func=llm_func)


# --- VM-02 / fake rows -------------------------------------------------------------


def test_the_fake_row_inserters_are_gone():
    for name in (
        "_insert_minimal_test_entities",
        "_insert_minimal_test_data",
        "_get_table_create_sql",
        "_get_table_indexes",
    ):
        assert not hasattr(SetupOrchestrator, name), name
    source = inspect.getsource(orchestrator_module)
    assert "test_entity_1" not in source
    assert "auto_setup_test" not in source


def test_entity_extraction_without_an_llm_fails_loudly_and_writes_nothing():
    cursor = Recording()
    orchestrator = orchestrator_over(cursor, llm_func=None)
    with pytest.raises(SetupError, match="llm_func"):
        orchestrator._perform_entity_extraction()
    assert not [s for s in cursor.statements if s.upper().startswith("INSERT")]


def test_entity_extraction_builds_the_service_the_way_it_is_declared():
    """`EntityExtractionService(storage_adapter=...)` raised TypeError (VM-02)."""
    from iris_vector_rag.services.entity_extraction import EntityExtractionService

    parameters = inspect.signature(EntityExtractionService.__init__).parameters
    assert "storage_adapter" not in parameters
    assert "extraction_method" not in parameters
    source = inspect.getsource(SetupOrchestrator._perform_entity_extraction)
    assert "storage_adapter=" not in source
    assert "extraction_method=" not in source


# --- VM-12: an unverifiable or empty result is not a success -------------------


def test_zero_embeddings_after_generation_is_an_error():
    cursor = Recording(ones=[(5, 0)])
    with pytest.raises(SetupError, match="no document embeddings"):
        orchestrator_over(cursor)._validate_embeddings_after_generation(
            "RAG.SourceDocuments", "embedding", "document"
        )


def test_no_rows_to_embed_is_an_error():
    cursor = Recording(ones=[(0, 0)])
    with pytest.raises(SetupError, match="no rows"):
        orchestrator_over(cursor)._validate_embeddings_after_generation(
            "RAG.SourceDocuments", "embedding", "document"
        )


def test_incomplete_embeddings_are_an_error_below_the_validators_threshold():
    cursor = Recording(ones=[(100, 50)])
    with pytest.raises(SetupError, match="50/100"):
        orchestrator_over(cursor)._validate_embeddings_after_generation(
            "RAG.SourceDocuments", "embedding", "document"
        )


def test_complete_embeddings_pass():
    cursor = Recording(ones=[(10, 10), ("0.1,0.2,0.3",)])
    orchestrator_over(cursor)._validate_embeddings_after_generation(
        "RAG.SourceDocuments", "embedding", "document"
    )


def test_a_string_formatted_embedding_is_an_error():
    cursor = Recording(ones=[(10, 10), ("[0.1,0.2]",)])
    with pytest.raises(SetupError, match="string"):
        orchestrator_over(cursor)._validate_embeddings_after_generation(
            "RAG.SourceDocuments", "embedding", "document"
        )


def test_a_query_error_while_validating_embeddings_propagates():
    cursor = Recording(fail_on=["COUNT"])
    with pytest.raises(SetupError, match="could not verify"):
        orchestrator_over(cursor)._validate_embeddings_after_generation(
            "RAG.SourceDocuments", "embedding", "document"
        )


# --- VM-15: schema honoured, no print, no second account of the catalog --------


def test_a_failed_existence_check_is_not_reported_as_absence():
    orchestrator = orchestrator_over(Recording())
    orchestrator._catalog_for = lambda: CatalogDouble(unreadable=True)
    with pytest.raises(SetupError, match="could not check"):
        orchestrator._table_exists("RAG", "SourceDocuments")


def test_the_existence_check_asks_about_the_requirements_schema():
    orchestrator = orchestrator_over(Recording())
    catalog = CatalogDouble(present=["rdf_labels"])
    orchestrator._catalog_for = lambda: catalog
    assert orchestrator._table_exists("SQLUSER", "rdf_labels") is True
    assert catalog.asked == [("SQLUSER", "rdf_labels")]


def test_no_print_in_library_code():
    tree = ast.parse(inspect.getsource(orchestrator_module))
    prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert prints == []


def test_the_orchestrator_has_no_inline_standard_ddl_or_fixed_vector_width():
    source = inspect.getsource(orchestrator_module)
    assert "CREATE TABLE" not in source
    assert not re.search(r"(?<!TO_)VECTOR\(", source)
    assert "INFORMATION_SCHEMA" not in source


def test_a_table_the_schema_manager_does_not_own_is_not_created_with_guessed_ddl():
    orchestrator = orchestrator_over(Recording())
    with pytest.raises(SetupError, match="rdf_labels"):
        orchestrator._create_table("SQLUSER", "rdf_labels")


def test_a_required_flag_false_table_is_not_created_by_setup():
    orchestrator = orchestrator_over(Recording())
    orchestrator._table_exists = Mock(return_value=False)
    orchestrator._create_table = Mock(return_value=True)
    orchestrator._fulfill_table_requirement(
        TableRequirement("Communities", "RAG", "optional", required=False)
    )
    orchestrator._create_table.assert_not_called()


def test_a_failed_table_creation_is_recorded_as_a_setup_error():
    orchestrator = orchestrator_over(Recording())
    orchestrator._table_exists = Mock(return_value=False)
    orchestrator._create_table = Mock(side_effect=SetupError("no ddl"))
    orchestrator._fulfill_table_requirement(
        TableRequirement("SourceDocuments", "RAG", "docs")
    )
    assert any("no ddl" in error for error in orchestrator.setup_errors)


# --- the token healer wrote a different table's columns -------------------------


def test_heal_token_embeddings_says_it_is_not_implemented():
    with pytest.raises(NotImplementedError, match="ColBERT"):
        orchestrator_over(Recording()).heal_token_embeddings()


# --- VM-09: the factory builds what the registry lists --------------------------


@pytest.mark.parametrize("pipeline_type", ["graphrag", "hybrid_graphrag"])
def test_graph_pipelines_are_built_with_the_schema_manager(pipeline_type):
    factory = factory_module.ValidatedPipelineFactory.__new__(
        factory_module.ValidatedPipelineFactory
    )
    factory.connection_manager = Mock()
    factory.config_manager = Mock()
    factory.logger = Mock()
    schema_manager = MagicMock()
    with (
        patch(
            "iris_vector_rag.storage.schema_manager.SchemaManager.create_schema_manager",
            return_value=schema_manager,
        ),
        patch.object(factory_module, "HybridGraphRAGPipeline") as pipeline,
    ):
        factory._create_pipeline_instance(pipeline_type, llm_func=None)
    assert pipeline.call_args.kwargs["schema_manager"] is schema_manager


def test_every_registered_pipeline_type_is_listed_as_available():
    factory = factory_module.ValidatedPipelineFactory.__new__(
        factory_module.ValidatedPipelineFactory
    )
    factory.validate_pipeline_type = lambda name: {"pipeline_type": name, "valid": True}
    assert set(factory.list_available_pipelines()) == set(
        PIPELINE_REQUIREMENTS_REGISTRY
    )


def test_an_unknown_pipeline_type_lists_the_registered_ones():
    factory = factory_module.ValidatedPipelineFactory.__new__(
        factory_module.ValidatedPipelineFactory
    )
    factory.connection_manager = Mock()
    factory.config_manager = Mock()
    with patch(
        "iris_vector_rag.storage.schema_manager.SchemaManager.create_schema_manager",
        return_value=MagicMock(),
    ):
        with pytest.raises(ValueError) as caught:
            factory._create_pipeline_instance("nonsense", llm_func=None)
    for name in PIPELINE_REQUIREMENTS_REGISTRY:
        assert name in str(caught.value)


# --- the migrate path for Entities lost `confidence` -----------------------------


def test_migrating_entities_keeps_the_confidence_column_the_adapter_writes():
    from iris_vector_rag.storage.schema_manager import SchemaManager

    manager = SchemaManager.__new__(SchemaManager)
    manager.schema_prefix = "RAG"
    executed: list[str] = []
    cursor = Mock()
    cursor.execute.side_effect = lambda sql, *a: executed.append(" ".join(sql.split()))
    with patch.object(SchemaManager, "_update_schema_metadata", create=True):
        manager._migrate_entities_table(
            cursor, {"vector_dimension": 384, "vector_data_type": "FLOAT"}, False
        )
    ddl = [s for s in executed if s.startswith("CREATE TABLE")]
    assert ddl and "confidence DOUBLE" in ddl[0], executed
