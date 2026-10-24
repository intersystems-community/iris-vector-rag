"""Bring your own table, against a real IRIS (feature 092).

MedTeam's and Štafeta's chunk tables are rebuilt with synthetic rows
(`tests/fixtures/byot/teams.py`), and the library's answer is compared with the answer
each team's own SQL gives on the same rows. The tables live in schemas of their own
(`BYOTMed`, `BYOTStf`), created and dropped around each test. Every test also checks
that the library left both tables exactly as it found them (SC-004).
"""

from __future__ import annotations

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import AttachedTableReadOnly, DimensionMismatchError
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.byot import teams as T

pytestmark = pytest.mark.integration

MEDTEAM = {
    "table_name": T.MED,
    "id_col": "EvidenceId",
    "text_col": "Text",
    "embedding_col": "Embedding",
    "title_col": "Title",
    "parent_col": "ResourceId",
}
STAFETA = {
    "table_name": T.STF,
    "id_col": ["DocRef", "Seq"],
    "text_col": "Text",
    "embedding_col": "Embedding",
    "parent_col": "DocRef",
}


@pytest.fixture
def byot_tables(iris_connection):
    """Both team tables, built before the test and dropped after; unchanged by it."""
    T.create(iris_connection)
    before = T.snapshot(iris_connection)
    yield iris_connection
    after = T.snapshot(iris_connection)
    T.drop(iris_connection)
    assert after == before, "the library changed an attached table"


def store_for(settings, **kwargs) -> IRISVectorStore:
    config = ConfigurationManager()
    return IRISVectorStore(ConnectionManager(config), config, **settings, **kwargs)


def ranked(results):
    return [(doc.id, round(score, 6)) for doc, score in results]


# --- US1: vector search over an attached table --------------------------------------


def test_us1_medteam_matches_the_teams_cosine_ranking(byot_tables):
    # Unscoped here; the patient and model scope is US2's test.
    cursor = byot_tables.cursor()
    try:
        cursor.execute(
            f"SELECT TOP 5 EvidenceId, VECTOR_COSINE(Embedding, TO_VECTOR(?, DOUBLE)) "
            f"AS Score FROM {T.MED} ORDER BY Score DESC",
            [T.vlit(T.MED_QUERY)],
        )
        expected = [(r[0], round(float(r[1]), 6)) for r in cursor.fetchall()]
    finally:
        cursor.close()

    store = store_for(MEDTEAM)
    results = store.similarity_search_by_embedding(T.MED_QUERY, 5)

    assert ranked(results) == expected
    doc = results[0][0]
    assert doc.metadata["parent_id"] == doc.id.rpartition("#")[0]
    assert doc.metadata["title"].startswith("Zpráva ")


def test_us1_stafeta_composite_ids(byot_tables):
    store = store_for(STAFETA)
    results = store.similarity_search_by_embedding(T.STF_QUERY, 3)
    ids = [doc.id for doc, _ in results]
    assert ids[0] == "DocumentReference/11#3"
    assert all(i.startswith("DocumentReference/") and "#" in i for i in ids)
    fetched = store.fetch_documents_by_ids(ids[:2])
    assert {d.id for d in fetched} == set(ids[:2])


def test_us1_a_query_of_the_wrong_width_is_refused(byot_tables):
    store = store_for(MEDTEAM)
    with pytest.raises(DimensionMismatchError):
        store.similarity_search_by_embedding([0.1] * 1536, 5)


def test_us1_writes_are_refused(byot_tables):
    from iris_vector_rag.core.models import Document

    store = store_for(MEDTEAM)
    with pytest.raises(AttachedTableReadOnly):
        store.add_documents([Document(id="x", page_content="t")])
    with pytest.raises(AttachedTableReadOnly):
        store.delete_documents(["P02-doc03#1"])


# --- US2: every search scoped by real-column filters ----------------------------------

MED_SCOPED = {
    **MEDTEAM,
    "filter_columns": ["PatientId", "EmbeddingModel", "DocDate"],
    "required_filters": ["PatientId", "EmbeddingModel"],
}
STF_SCOPED = {
    **STAFETA,
    "filter_columns": ["PatientRef", "EmbeddingModel", "AuthoredOn", "ReceivedOn"],
    "required_filters": ["PatientRef"],
}
STF_WHERE = {
    "PatientRef": "Patient/2",
    "EmbeddingModel": T.STF_MODEL,
    "AuthoredOn": ("<", T.AS_OF_AUTHORED_BEFORE),
    "ReceivedOn": ("<=", T.AS_OF_RECEIVED_UNTIL),
}


@pytest.mark.parametrize("model", T.MODELS)
def test_us2_medteam_patient_and_model_match_the_teams_sql(byot_tables, model):
    cursor = byot_tables.cursor()
    try:
        expected = T.med_team_topk(cursor, model)
    finally:
        cursor.close()

    store = store_for(MED_SCOPED)
    results = store.similarity_search_by_embedding(
        T.MED_QUERY, 5, where={"PatientId": "P02", "EmbeddingModel": model}
    )

    assert ranked(results) == expected


def test_us2_stafeta_as_of_rule_matches_the_teams_sql(byot_tables):
    cursor = byot_tables.cursor()
    try:
        expected = T.stf_team_vector_topk(cursor)
        unscoped = T.stf_team_vector_topk(cursor, as_of=False)
    finally:
        cursor.close()
    assert expected != unscoped  # the as-of rule bites on these rows

    store = store_for(STF_SCOPED)
    results = store.similarity_search_by_embedding(T.STF_QUERY, 5, where=STF_WHERE)

    assert ranked(results) == expected


def test_us2_a_list_is_in(byot_tables):
    store = store_for(MED_SCOPED)
    results = store.similarity_search_by_embedding(
        T.MED_QUERY,
        10,
        where={"PatientId": ["P01", "P03"], "EmbeddingModel": T.MODELS[0]},
    )
    assert results and all(doc.id[:3] in ("P01", "P03") for doc, _ in results)


def test_us2_a_missing_required_filter_is_refused(byot_tables):
    from iris_vector_rag.exceptions import RequiredFilterMissing

    store = store_for(MED_SCOPED)
    with pytest.raises(RequiredFilterMissing, match="EmbeddingModel"):
        store.similarity_search_by_embedding(T.MED_QUERY, 5, where={"PatientId": "P02"})


def test_us2_an_injection_value_matches_nothing(byot_tables):
    store = store_for(MED_SCOPED)
    results = store.similarity_search_by_embedding(
        T.MED_QUERY,
        5,
        where={"PatientId": "P02' OR 1=1 --", "EmbeddingModel": T.MODELS[0]},
    )
    assert results == []


# --- US3: a ready-made pipeline from configuration alone -------------------------------


def test_us3_basic_pipeline_from_configuration(
    byot_tables, monkeypatch, use_test_embedder
):
    for key, value in {
        "TABLE_NAME": T.MED,
        "ID_COL": "EvidenceId",
        "TEXT_COL": "Text",
        "EMBEDDING_COL": "Embedding",
        "FILTER_COLUMNS": "PatientId,EmbeddingModel",
        "REQUIRED_FILTERS": "PatientId,EmbeddingModel",
    }.items():
        monkeypatch.setenv(f"RAG_STORAGE__IRIS__{key}", value)
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    config = ConfigurationManager()
    pipeline = BasicRAGPipeline(
        ConnectionManager(config), config, llm_func=lambda p: "ok"
    )

    result = pipeline.query(
        "apixaban vysazen",
        top_k=4,
        where={"PatientId": "P03", "EmbeddingModel": T.MODELS[1]},
    )

    docs = result["retrieved_documents"]
    assert len(docs) == 4
    assert all(doc.id.startswith("P03-") for doc in docs)
    assert all(doc.page_content for doc in docs)

    from iris_vector_rag.exceptions import RequiredFilterMissing

    with pytest.raises(RequiredFilterMissing):
        pipeline.query("apixaban", top_k=4, where={"PatientId": "P03"})


# --- US4: the text leg on the table's own index -------------------------------------

STF_TEXT = {
    **STF_SCOPED,
    "text_index": "DocChunkText",
    "text_language": "cs",
    "text_terms": "prefix",
}


def test_us4_stafeta_text_leg_matches_the_teams_ifind(byot_tables):
    cursor = byot_tables.cursor()
    try:
        expected = T.stf_team_ifind(cursor)
        unscoped = T.stf_team_ifind(cursor, as_of=False)
    finally:
        cursor.close()
    assert expected != unscoped  # the as-of rule bites on the text leg too

    store = store_for(STF_TEXT)
    docs = store.text_search(T.STF_TEXT_QUERY, top_k=50, where=STF_WHERE)

    assert sorted(d.id for d in docs) == expected


def test_us4_rrf_keeps_every_leg_in_scope(byot_tables, use_test_embedder):
    from iris_vector_rag.core.query_options import QueryOptions
    from iris_vector_rag.retrieval.engine import RetrievalEngine

    store = store_for(STF_TEXT)
    engine = RetrievalEngine(vector_store=store, connection=None)
    docs = engine.retrieve(
        QueryOptions(query=T.STF_TEXT_QUERY, retrieval="rrf", top_k=10, where=STF_WHERE)
    )

    allowed = set(T.stf_team_ifind(byot_tables.cursor())) | {
        doc_id for doc_id, _ in T.stf_team_vector_topk(byot_tables.cursor(), k=100)
    }
    assert docs and {d.id for d in docs} <= allowed


def test_us4_an_attached_table_without_its_index_named_is_refused(byot_tables):
    from iris_vector_rag.exceptions import TextIndexMissingError

    store = store_for(STF_SCOPED)
    with pytest.raises(TextIndexMissingError, match="text_index"):
        store.text_search(T.STF_TEXT_QUERY, where=STF_WHERE)


# --- US5: the similarity measure and the table's HNSW index ---------------------------


def test_us5_a_measure_the_hnsw_index_was_not_built_for_is_warned_about(
    byot_tables, caplog
):
    import logging

    with caplog.at_level(logging.WARNING):
        store_for(STAFETA, metric="dot").similarity_search_by_embedding(T.STF_QUERY, 3)
    warnings = [r.getMessage() for r in caplog.records if "HNSW" in r.getMessage()]
    assert len(warnings) == 1 and "DocChunkHNSW" in warnings[0]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        store_for(STAFETA).similarity_search_by_embedding(T.STF_QUERY, 3)
    assert not [r for r in caplog.records if "HNSW" in r.getMessage()]


def test_us5_cosine_and_dot_rank_non_unit_vectors_differently(iris_connection):
    """Why the default matters: vectors that are not unit length (Gemini at reduced
    dimensions) rank differently under the two measures."""
    table = "BYOTMetric.Chunk"
    cursor = iris_connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(
            f"CREATE TABLE {table} (ChunkId VARCHAR(20), Body VARCHAR(200), "
            "Vec VECTOR(DOUBLE, 2))"
        )
        # a: same direction as the query, short; b: off direction, long.
        for chunk_id, vec in (("a", "[1,0]"), ("b", "[6,3]")):
            cursor.execute(
                f"INSERT INTO {table} VALUES (?, ?, TO_VECTOR(?, DOUBLE, 2))",
                [chunk_id, chunk_id, vec],
            )
        iris_connection.commit()
        settings = {
            "table_name": table,
            "id_col": "ChunkId",
            "text_col": "Body",
            "embedding_col": "Vec",
        }
        cosine = store_for(settings).similarity_search_by_embedding([1.0, 0.0], 2)
        dot = store_for(settings, metric="dot").similarity_search_by_embedding(
            [1.0, 0.0], 2
        )
        assert [d.id for d, _ in cosine] == ["a", "b"]
        assert [d.id for d, _ in dot] == ["b", "a"]
    finally:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        iris_connection.commit()
        cursor.close()


# --- review fix: an id that repeats outside the scope --------------------------------


def test_text_leg_returns_the_in_scope_row_of_a_repeated_id(iris_connection):
    """MedTeam's EvidenceId repeats once per embedding model. The text leg must
    present the row of the model the search is scoped to, not whichever row of
    that id the database returns last (FR-010)."""
    table = "BYOTDup.Chunk"
    cursor = iris_connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(
            f"CREATE TABLE {table} (EvidenceId VARCHAR(20), EmbeddingModel VARCHAR(20), "
            "Body VARCHAR(200), Vec VECTOR(DOUBLE, 2))"
        )
        for model, body in (("m1", "alpha apple"), ("m2", "alpha banana")):
            cursor.execute(
                f"INSERT INTO {table} VALUES (?, ?, ?, TO_VECTOR('[1,0]', DOUBLE, 2))",
                ["e1", model, body],
            )
        cursor.execute(
            f"CREATE INDEX DupText ON {table} (Body) "
            "AS %iFind.Index.Basic(LANGUAGE='en', LOWER=1)"
        )
        iris_connection.commit()
        settings = {
            "table_name": table,
            "id_col": "EvidenceId",
            "text_col": "Body",
            "embedding_col": "Vec",
            "filter_columns": ["EmbeddingModel"],
            "required_filters": ["EmbeddingModel"],
            "text_index": "DupText",
        }
        for model, body in (("m1", "alpha apple"), ("m2", "alpha banana")):
            docs = store_for(settings).text_search(
                "alpha", top_k=5, where={"EmbeddingModel": model}
            )
            assert [(d.id, d.page_content) for d in docs] == [("e1", body)]
    finally:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        iris_connection.commit()
        cursor.close()
