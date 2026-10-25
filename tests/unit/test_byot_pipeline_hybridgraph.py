"""`HybridGraphRAGPipeline.query(..., where=...)` (feature 092, FR-021).

Every retrieval path honours `where`: the ones that read the document table filter in
SQL, and a document a graph leg names is fetched under the same predicate, so a hit
the scope excludes never reaches the result. A path that cannot apply `where` raises;
a required filter cannot be left out.

The corpus is two patients' chunks (BYOTMed.EvidenceChunk, as MedTeam's catalog
describes it) and an entity graph over them. The cursor answers like the database
would: it applies the `"PatientId" = ?` predicate the pipeline sends, so a query that
forgot the predicate returns the other patient's row and the test fails.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import FilterError, RequiredFilterMissing
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import HybridRetrievalMethods
from tests.unit.test_byot_store_vector import MEDTEAM, QUERY, team_store

SCOPED = {**MEDTEAM, "filter_columns": ["PatientId"], "required_filters": ["PatientId"]}
OPEN = {**MEDTEAM, "filter_columns": ["PatientId"]}
WHERE = {"PatientId": "P02"}

#: doc id -> (patient, text, title, parent)
DOCS = {
    "p2-a": ("P02", "P02 apixaban stopped", "Zpráva P02", "r2"),
    "p2-b": ("P02", "P02 biopsy done", "Zpráva P02b", "r2"),
    "p3-a": ("P03", "P03 secret note", "Zpráva P03", "r3"),
}
#: entity id -> the document it was extracted from
ENTITIES = {"e-in": "p2-a", "e-in2": "p2-b", "e-out": "p3-a"}
IN_SCOPE_TEXTS = {"P02 apixaban stopped", "P02 biopsy done"}
METHODS = ["hybrid", "rrf", "text", "vector", "kg"]
PREDICATE = '"PatientId" = ?'


class CorpusCursor:
    """The corpus above behind the statements the pipeline sends; records them all."""

    rowcount = 1

    def __init__(self):
        self.executed: list = []
        self.description = None
        self._result: list = []

    def execute(self, sql, params=None):
        self.executed.append((sql, list(params or [])))
        upper = " ".join(str(sql).split()).upper()
        params = list(params or [])
        self.description = None
        scoped = PREDICATE.upper() in upper
        patient = params[-1] if scoped else None
        if "INFORMATION_SCHEMA" in upper or "%DICTIONARY" in upper:
            raise AssertionError(f"read the catalog itself: {sql}")
        if "SELECT DISTINCT SOURCE_DOC_ID" in upper:
            self.description = (("source_doc_id",),)
            self._result = sorted(
                {(ENTITIES[e],) for e in params if e in ENTITIES}, key=str
            )
        elif ".ENTITIES" in upper and "SELECT TOP 1" in upper:
            entity = params[0]
            doc = DOCS.get(ENTITIES.get(entity))
            if doc and (not scoped or doc[0] == patient):
                self._result = [(doc[1],)]
            else:
                self._result = []
        elif "VECTOR_COSINE" in upper or "VECTOR_DOT_PRODUCT" in upper:
            self._result = [
                (doc_id, text, title, parent, 0.9)
                for doc_id, (who, text, title, parent) in DOCS.items()
                if not scoped or who == patient
            ]
        elif " IN (" in upper:
            ids = params[:-1] if scoped else params
            self._result = [
                (doc_id, text, title, parent)
                for doc_id, (who, text, title, parent) in DOCS.items()
                if doc_id in ids and (not scoped or who == patient)
            ]
        else:
            self._result = []

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None

    def close(self):
        pass

    def reading(self, table: str = "BYOTMED.EVIDENCECHUNK"):
        """Statements that read the document table."""
        return [
            (sql, params)
            for sql, params in self.executed
            if table in " ".join(str(sql).split()).upper()
        ]


class Engine:
    """The graph engine: it names entities, never documents."""

    def __init__(self, entities):
        self.entities = list(entities)
        self.asked: list = []

    def kg_RRF_FUSE(self, **kwargs):
        self.asked.append(kwargs)
        return [(e, 0.5, 0.4, 0.3) for e in self.entities]

    def kg_TXT(self, **kwargs):
        self.asked.append(kwargs)
        return [(e, 0.9) for e in self.entities]

    def kg_KNN_VEC(self, **kwargs):
        self.asked.append(kwargs)
        return [(e, 0.8) for e in self.entities]

    def multi_modal_search(self, **kwargs):
        self.asked.append(kwargs)
        return [{"entity_id": e, "fusion_score": 0.7} for e in self.entities]


class Embedder:
    def embed_text(self, text):
        return list(QUERY)

    def get_embeddings(self, texts):
        return [list(QUERY) for _ in texts]


def pipeline_for(settings, cursor, engine=None) -> HybridGraphRAGPipeline:
    """A pipeline over the corpus; with an ``engine``, the iris_vector_graph legs run."""
    store = team_store(settings, cursor)
    connection = MagicMock()
    connection.cursor.return_value = cursor
    connections = MagicMock(spec=ConnectionManager)
    connections.get_connection.return_value = connection

    pipeline = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    pipeline.vector_store = store
    pipeline.connection_manager = connections
    pipeline.embedding_manager = Embedder()
    pipeline.llm_func = None
    pipeline.default_top_k = 5
    pipeline.max_depth = 1
    pipeline._executor = None
    pipeline.schema_manager = None
    pipeline.iris_engine = engine
    pipeline.fusion_engine = engine
    pipeline.text_engine = None
    pipeline.vector_optimizer = None
    pipeline.retrieval_methods = (
        HybridRetrievalMethods(engine, engine, None, None, pipeline.embedding_manager)
        if engine is not None
        else None
    )
    # No LLM names the query's entities; the graph is read from these.
    pipeline._find_seed_entities = lambda query: ["e-in", "e-out"]
    pipeline._expand_neighborhood = lambda nodes, depth: set(nodes)
    return pipeline


def run(pipeline, method, where=WHERE, **kwargs):
    extra = {} if where is None else {"where": where}
    return pipeline.query(
        "apixaban", method=method, top_k=5, generate_answer=False, **extra, **kwargs
    )


def texts(result):
    return {doc.page_content for doc in result["retrieved_documents"]}


def assert_every_read_is_scoped(cursor):
    reads = cursor.reading()
    assert reads, "the document table was never read"
    for sql, params in reads:
        assert PREDICATE in sql, f"unscoped read of the document table: {sql}"
        assert "P02" in params


# --- legs through the graph engine ---------------------------------------------------


@pytest.mark.parametrize(
    "method,label",
    [
        ("hybrid", "hybrid_fusion"),
        ("rrf", "rrf_fusion"),
        ("text", "enhanced_text"),
        ("vector", "hnsw_vector"),
    ],
)
def test_a_document_a_graph_leg_names_outside_the_scope_is_dropped(method, label):
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor, Engine(["e-in", "e-out", "e-in2"]))

    result = run(pipeline, method)

    assert result["metadata"]["retrieval_method"] == label
    assert texts(result) == IN_SCOPE_TEXTS
    assert_every_read_is_scoped(cursor)


@pytest.mark.parametrize("method", ["hybrid", "rrf", "text", "vector"])
def test_a_graph_leg_that_finds_only_out_of_scope_documents_falls_back_in_scope(method):
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor, Engine(["e-out"]))

    result = run(pipeline, method)

    assert result["metadata"]["retrieval_method"] == "vector_fallback"
    assert texts(result) == IN_SCOPE_TEXTS
    assert_every_read_is_scoped(cursor)


def test_a_scoped_graph_leg_asks_for_more_than_top_k_so_the_scope_does_not_starve_it():
    engine = Engine(["e-in"])
    pipeline = pipeline_for(SCOPED, CorpusCursor(), engine)

    run(pipeline, "rrf")

    assert engine.asked[0]["k"] > 5


def test_without_a_scope_the_graph_leg_and_its_content_query_are_unchanged():
    cursor = CorpusCursor()
    engine = Engine(["e-in", "e-out"])
    pipeline = pipeline_for(OPEN, cursor, engine)

    result = run(pipeline, "rrf", where=None)

    assert engine.asked[0]["k"] == 5
    assert texts(result) == {"P02 apixaban stopped", "P03 secret note"}
    (sql, _), *_ = cursor.reading()
    assert " ".join(sql.split()).startswith(
        "SELECT TOP 1 s.Text FROM BYOTMed.EvidenceChunk s JOIN BYOTMed.Entities e "
        "ON e.source_doc_id = s.EvidenceId WHERE e.entity_id = ?"
    )
    assert PREDICATE not in sql


# --- legs without the graph engine ---------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_every_method_without_the_engine_stays_in_scope(method):
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor)

    result = run(pipeline, method)

    assert result["retrieved_documents"]
    assert texts(result) <= IN_SCOPE_TEXTS
    assert_every_read_is_scoped(cursor)


@pytest.mark.parametrize(
    "strategy", ["kg_primary", "kg_vector_hybrid", "vector_primary"]
)
@pytest.mark.parametrize("method", ["hybrid", "rrf"])
def test_every_hybrid_strategy_stays_in_scope(method, strategy):
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor)
    pipeline._analyze_query_for_hybrid_strategy = lambda q: {
        "recommended_strategy": strategy
    }

    result = run(pipeline, method)

    assert result["retrieved_documents"]
    assert texts(result) <= IN_SCOPE_TEXTS
    assert_every_read_is_scoped(cursor)


def test_the_entity_graph_leg_hydrates_documents_under_the_predicate():
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor)

    result = run(pipeline, "kg")

    assert result["metadata"]["retrieval_method"] == "knowledge_graph"
    assert texts(result) == {"P02 apixaban stopped"}
    ((sql, params),) = cursor.reading()
    assert PREDICATE in sql and params[-1] == "P02"


def test_the_entity_guided_text_and_vector_legs_use_the_real_entity_ids():
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor)

    run(pipeline, "text")
    run(pipeline, "vector")

    asked = [p for sql, p in cursor.executed if "SOURCE_DOC_ID" in sql.upper()]
    assert asked and all(set(p) == {"e-in", "e-out"} for p in asked)


# --- refusing what cannot be applied --------------------------------------------------


@pytest.mark.parametrize(
    "engine", [None, Engine(["e-in"])], ids=["no-engine", "engine"]
)
@pytest.mark.parametrize("method", METHODS)
def test_a_required_filter_cannot_be_left_out(method, engine):
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor, engine)

    with pytest.raises(RequiredFilterMissing, match="PatientId"):
        run(pipeline, method, where=None)

    assert cursor.reading() == []
    if engine is not None:
        assert engine.asked == []


@pytest.mark.parametrize(
    "engine", [None, Engine(["e-in"])], ids=["no-engine", "engine"]
)
@pytest.mark.parametrize("method", METHODS)
def test_a_filter_on_an_undeclared_column_is_refused_not_dropped(method, engine):
    cursor = CorpusCursor()
    pipeline = pipeline_for(SCOPED, cursor, engine)

    with pytest.raises(FilterError, match="Department"):
        run(pipeline, method, where={**WHERE, "Department": "x"})

    assert cursor.reading() == []
    if engine is not None:
        assert engine.asked == []


def test_where_on_a_pipeline_with_no_vector_store_is_refused():
    pipeline = pipeline_for(SCOPED, CorpusCursor())
    pipeline.vector_store = None

    with pytest.raises(FilterError):
        run(pipeline, "vector")
