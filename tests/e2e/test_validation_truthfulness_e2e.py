"""Validation and auto-setup against a real IRIS report what is true.

One story each, on the standard RAG tables of this test session's own database:

1. Validation of an empty database says the tables are missing, and says a failed
   count is a failed count.
2. Setup of a document pipeline creates the canonical tables, embeds the documents
   that are there, chunks them in the canonical layout, and leaves no row it invented.
3. Setup of a graph pipeline extracts entities from the documents' text (not their
   names), is valid without the optional `Communities` table, and says plainly, with
   no fake rows, when it has no LLM.
4. Migrating `Entities` keeps the `confidence` column the storage adapter writes.

The embedder is the real sentence-transformer and the LLM is OpenAI's (OPENAI_API_KEY from
the environment), wrapped only to record the prompts it is sent.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.validation.orchestrator import SetupOrchestrator
from iris_vector_rag.validation.requirements import get_pipeline_requirements
from iris_vector_rag.validation.validator import PreConditionValidator
from tests.support.real_services import real_llm

RAG_TABLES = [
    "DocumentChunks",
    "EntityRelationships",
    "Entities",
    "Communities",
    "VectorEmbeddings",
    "SourceDocuments",
]

DOCUMENTS = {
    "doc-1": "Warfarin dosing needs monitoring of the INR. Aspirin raises bleeding risk.",
    "doc-2": "Metformin is first line for type two diabetes. Insulin is added later.",
}


def _drop_rag_tables(connection_manager: ConnectionManager) -> None:
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    try:
        for table in RAG_TABLES:
            cursor.execute(f"DROP TABLE IF EXISTS RAG.{table}")
        connection.commit()
    finally:
        cursor.close()


def _scalar(connection_manager: ConnectionManager, sql: str, params=None):
    cursor = connection_manager.get_connection().cursor()
    try:
        cursor.execute(sql, params or [])
        return cursor.fetchone()[0]
    finally:
        cursor.close()


@pytest.fixture
def managers():
    """A config and connection manager over an empty RAG schema."""
    config = ConfigurationManager()
    connections = ConnectionManager(config)
    _drop_rag_tables(connections)
    yield config, connections
    _drop_rag_tables(connections)


@pytest.fixture
def loaded(managers):
    """The same, with the canonical document table holding two un-embedded documents."""
    config, connections = managers
    assert SchemaManager(connections, config).ensure_table_schema("SourceDocuments")
    connection = connections.get_connection()
    cursor = connection.cursor()
    try:
        for doc_id, text in DOCUMENTS.items():
            cursor.execute(
                "INSERT INTO RAG.SourceDocuments (doc_id, title, text_content) "
                "VALUES (?, ?, ?)",
                [doc_id, f"Title of {doc_id}", text],
            )
        connection.commit()
    finally:
        cursor.close()
    return config, connections


def _llm(prompts: list):
    """The real LLM, recording every prompt it is sent (so a test can say what it saw)."""
    real = real_llm()

    def call(prompt: str) -> str:
        prompts.append(prompt)
        return real(prompt)

    return call


# --- story 1: validation of an empty database ---------------------------------------


@pytest.mark.e2e
@pytest.mark.requires_database
class TestValidationOfAnEmptyDatabase:
    def test_missing_tables_are_reported_as_missing(self, managers):
        _, connections = managers
        report = PreConditionValidator(connections).validate_pipeline_requirements(
            get_pipeline_requirements("graphrag")
        )

        assert not report.overall_valid
        for name in ("SourceDocuments", "Entities", "EntityRelationships"):
            result = report.table_validations[name]
            assert result.details["error_kind"] == "missing_table", result
            assert "does not exist" in result.message
            assert f"Create table RAG.{name}" in result.suggestions
        # Listed with the required tables but flagged not required: reported, not gating.
        assert "Communities_optional" in report.table_validations
        assert "Communities" not in report.summary

    def test_a_failed_count_on_a_table_that_exists_is_not_a_missing_table(self, loaded):
        config, connections = loaded

        class FailingCount:
            def __init__(self, inner):
                self.inner = inner

            def execute(self, sql, *args):
                if "COUNT(*) FROM RAG.SourceDocuments" in sql:
                    raise RuntimeError("connection reset by peer")
                return self.inner.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self.inner, name)

        class Flaky:
            def get_connection(self):
                real = connections.get_connection()

                class Connection:
                    def cursor(self_inner):
                        return FailingCount(real.cursor())

                    def __getattr__(self_inner, name):
                        return getattr(real, name)

                return Connection()

        report = PreConditionValidator(Flaky()).validate_pipeline_requirements(
            get_pipeline_requirements("basic")
        )
        result = report.table_validations["SourceDocuments"]
        assert not result.is_valid
        assert result.details["error_kind"] == "count_failed", result
        assert result.details["table_exists"] is True
        assert "does not exist" not in result.message
        assert "Create table RAG.SourceDocuments" not in result.suggestions


# --- story 2: a document pipeline -----------------------------------------------------


@pytest.mark.e2e
@pytest.mark.requires_database
class TestDocumentPipelineSetup:
    def test_setup_on_an_empty_database_creates_tables_and_invents_no_rows(
        self, managers
    ):
        config, connections = managers
        orchestrator = SetupOrchestrator(connections, config)
        report = orchestrator.setup_pipeline("basic")

        # The table exists in its canonical layout; it holds nothing, and the report
        # says the minimum is unmet rather than meeting it with a placeholder.
        assert (
            _scalar(
                connections,
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE "
                "TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments' "
                "AND COLUMN_NAME = 'doc_id'",
            )
            == 1
        )
        assert _scalar(connections, "SELECT COUNT(*) FROM RAG.SourceDocuments") == 0
        assert not report.overall_valid
        assert report.table_validations["SourceDocuments"].details["error_kind"] == (
            "below_min_rows"
        )
        assert any(
            "SourceDocuments" in error for error in report.setup_errors
        ), report.setup_errors

    def test_setup_embeds_and_chunks_the_documents_that_are_there(self, loaded):
        config, connections = loaded
        orchestrator = SetupOrchestrator(connections, config)
        report = orchestrator.setup_pipeline("basic")

        assert report.overall_valid, (report.summary, report.setup_errors)
        assert report.setup_errors == []
        assert (
            _scalar(
                connections,
                "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NULL",
            )
            == 0
        )
        # Chunks in the canonical layout: the chunk's own key, its parent, its text.
        assert _scalar(connections, "SELECT COUNT(*) FROM RAG.DocumentChunks") == len(
            DOCUMENTS
        )
        chunk = _rows(
            connections,
            "SELECT chunk_text, chunk_embedding FROM RAG.DocumentChunks "
            "WHERE document_id = 'doc-1'",
        )
        assert len(chunk) == 1
        assert str(chunk[0][0]) == DOCUMENTS["doc-1"]
        assert chunk[0][1] is not None
        # And no row that was not loaded or derived from a loaded document.
        assert _scalar(connections, "SELECT COUNT(*) FROM RAG.SourceDocuments") == len(
            DOCUMENTS
        )
        assert report.embedding_validations["embedding"].is_valid
        assert report.embedding_validations["chunk_embeddings_optional"].is_valid


# --- story 3: a graph pipeline --------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.requires_database
class TestGraphPipelineSetup:
    def test_entities_are_extracted_from_the_text_and_communities_is_optional(
        self, loaded
    ):
        config, connections = loaded
        prompts: list = []
        orchestrator = SetupOrchestrator(connections, config, llm_func=_llm(prompts))
        report = orchestrator.setup_pipeline("graphrag")

        # The LLM saw the documents' text, not their names.
        assert any("Warfarin dosing" in prompt for prompt in prompts), prompts
        assert not any("Title of doc" in prompt for prompt in prompts)

        names = {
            row[0] for row in _rows(connections, "SELECT entity_name FROM RAG.Entities")
        }
        # The real model's capitalisation is its own; the drugs in the text are what count.
        lowered = {name.lower() for name in names}
        assert {"warfarin", "aspirin", "metformin", "insulin"} <= lowered, names
        assert names.isdisjoint({"Test Entity"})
        assert (
            _scalar(
                connections,
                "SELECT COUNT(*) FROM RAG.EntityRelationships "
                "WHERE relationship_type = 'co_occurrence'",
            )
            >= 2
        )
        assert (
            _scalar(
                connections,
                "SELECT COUNT(*) FROM RAG.Entities WHERE entity_id = 'test_entity_1'",
            )
            == 0
        )

        # Valid although Communities (required=False) holds nothing.
        assert report.setup_errors == [], report.setup_errors
        assert report.overall_valid, report.summary
        assert "Communities" not in {
            name for name in report.required_names if name.startswith("Communities")
        }

    def test_without_an_llm_nothing_is_invented_and_the_report_says_why(self, loaded):
        config, connections = loaded
        orchestrator = SetupOrchestrator(connections, config)
        report = orchestrator.setup_pipeline("hybrid_graphrag")

        assert _scalar(connections, "SELECT COUNT(*) FROM RAG.Entities") == 0
        assert _scalar(connections, "SELECT COUNT(*) FROM RAG.EntityRelationships") == 0
        assert not report.overall_valid
        assert not report.table_validations["Entities"].is_valid
        assert any(
            "llm_func" in error for error in report.setup_errors
        ), report.setup_errors


def _rows(connections, sql):
    cursor = connections.get_connection().cursor()
    try:
        cursor.execute(sql)
        return [tuple(row) for row in cursor.fetchall()]
    finally:
        cursor.close()


# --- story 4: migrating Entities ----------------------------------------------------


@pytest.mark.e2e
@pytest.mark.requires_database
def test_migrating_entities_keeps_the_confidence_column(managers):
    config, connections = managers
    manager = SchemaManager(connections, config)
    assert manager.ensure_table_schema("SourceDocuments")
    assert manager.migrate_table("Entities", preserve_data=False)

    columns = {
        row[0].lower()
        for row in _rows(
            connections,
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'Entities'",
        )
    }
    assert {"confidence", "entity_id", "source_doc_id"} <= columns


# --- story 5: the factory ---------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.requires_database
@pytest.mark.parametrize("pipeline_type", ["graphrag", "hybrid_graphrag"])
def test_factory_auto_setup_builds_the_graph_pipeline_it_set_up(loaded, pipeline_type):
    from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
    from iris_vector_rag.validation.factory import ValidatedPipelineFactory

    config, connections = loaded
    prompts: list = []
    factory = ValidatedPipelineFactory(connections, config)
    pipeline = factory.create_pipeline(
        pipeline_type, llm_func=_llm(prompts), auto_setup=True
    )

    assert isinstance(pipeline, HybridGraphRAGPipeline)
    assert pipeline.schema_manager is not None
    # The real model's entity list varies run to run (3 to 4+ entities for these two
    # documents); what is guaranteed is that each document got at least one.
    assert _scalar(
        connections, "SELECT COUNT(DISTINCT source_doc_id) FROM RAG.Entities"
    ) == len(DOCUMENTS)
    assert _scalar(connections, "SELECT COUNT(*) FROM RAG.Entities") >= len(DOCUMENTS)
    assert any("Warfarin dosing" in prompt for prompt in prompts)
    assert set(factory.list_available_pipelines()) >= {"graphrag", "hybrid_graphrag"}
