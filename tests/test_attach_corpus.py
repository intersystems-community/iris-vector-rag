import logging
import os

import numpy as np
import pytest

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "1972"))
TOKEN_DIM = 384

logger = logging.getLogger(__name__)


def _get_conn():
    import iris.dbapi as dbapi

    return dbapi.connect(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )


def _get_engine(conn):
    import intersystems_iris
    from iris_vector_graph import IRISGraphEngine

    native = intersystems_iris.createConnection(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )
    return IRISGraphEngine(native)


def _create_test_table(conn, table_name, n_rows, dim=TOKEN_DIM):
    cur = conn.cursor()
    try:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        cur.execute(
            f"CREATE TABLE {table_name} ("
            f"doc_id VARCHAR(50) PRIMARY KEY, "
            f"text_content VARCHAR(2000), "
            f"embedding VECTOR(DOUBLE, {dim}))"
        )
        conn.commit()
        rng = np.random.default_rng(42)
        for i in range(n_rows):
            vec = rng.random(dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            vec_str = ",".join(str(float(v)) for v in vec)
            cur.execute(
                f"INSERT INTO {table_name} (doc_id, text_content, embedding) "
                f"VALUES (?, ?, TO_VECTOR(?, DOUBLE, {dim}))",
                [f"doc_{i:04d}", f"Test document {i} about topic {i % 10}.", vec_str],
            )
        conn.commit()
    finally:
        cur.close()
    return n_rows


def _drop_test_table(conn, table_name):
    cur = conn.cursor()
    try:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
    except Exception:
        pass
    finally:
        cur.close()


def _make_pipeline(conn, engine):
    from unittest.mock import MagicMock

    from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

    pipeline = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    pipeline.iris_engine = engine
    pipeline._attached_corpora = {}
    pipeline.connection_manager = MagicMock()
    pipeline.config_manager = MagicMock()
    return pipeline


@pytest.fixture(scope="module")
def conn():
    c = _get_conn()
    yield c
    c.close()


@pytest.fixture(scope="module")
def engine(conn):
    import intersystems_iris

    native = intersystems_iris.createConnection(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )
    from iris_vector_graph import IRISGraphEngine

    e = IRISGraphEngine(native, embedding_dimension=TOKEN_DIM)
    e.initialize_schema(auto_deploy_objectscript=False)
    return e


@pytest.fixture(scope="module")
def pipeline(conn, engine):
    return _make_pipeline(conn, engine)


@pytest.fixture(scope="module")
def test_table(conn):
    table = "Test069.CorpusA"
    _create_test_table(conn, table, 100)
    yield table
    _drop_test_table(conn, table)


# --- US1: Attach RAG corpus ---


@pytest.mark.integration
class TestAttachCorpusBasic:
    def test_attach_returns_result(self, pipeline, test_table):
        result = pipeline.attach_existing_corpus(
            source_table=test_table,
            id_col="doc_id",
            text_col="text_content",
            embedding_col="embedding",
            graph_label="TestDoc",
        )
        assert result["table"] == test_table
        assert result["label"] == "TestDoc"
        assert result["dimension"] == TOKEN_DIM
        assert result["row_count"] == 100
        assert isinstance(result["has_hnsw_index"], (bool, type(None)))

    def test_graph_query_returns_rows(self, pipeline, engine, test_table):
        pipeline.attach_existing_corpus(
            source_table=test_table,
            id_col="doc_id",
            text_col="text_content",
            embedding_col="embedding",
            graph_label="TestDoc2",
        )
        results = engine.execute_cypher("MATCH (d:TestDoc2) RETURN d.doc_id LIMIT 5")
        assert len(results) > 0

    def test_vector_search_returns_results(self, pipeline, engine, test_table):
        pipeline.attach_existing_corpus(
            source_table=test_table,
            id_col="doc_id",
            text_col="text_content",
            embedding_col="embedding",
            graph_label="TestDocVS",
        )
        rng = np.random.default_rng(99)
        q = rng.random(TOKEN_DIM).astype(np.float32)
        q /= np.linalg.norm(q)
        results = engine.vector_search(
            test_table, "embedding", q.tolist(), top_k=5, id_col="doc_id"
        )
        assert len(results) > 0


@pytest.mark.integration
class TestAttachCorpusIdempotent:
    def test_double_attach_no_error(self, pipeline, test_table):
        r1 = pipeline.attach_existing_corpus(
            source_table=test_table,
            id_col="doc_id",
            text_col="text_content",
            embedding_col="embedding",
            graph_label="IdempotentLabel",
        )
        r2 = pipeline.attach_existing_corpus(
            source_table=test_table,
            id_col="doc_id",
            text_col="text_content",
            embedding_col="embedding",
            graph_label="IdempotentLabel",
        )
        assert r1["dimension"] == r2["dimension"]
        assert r1["label"] == r2["label"]


@pytest.mark.integration
class TestNewRowsVisible:
    def test_new_row_in_graph_after_insert(self, pipeline, conn, engine, test_table):
        pipeline.attach_existing_corpus(
            source_table=test_table,
            id_col="doc_id",
            text_col="text_content",
            embedding_col="embedding",
            graph_label="NewRowLabel",
        )
        cur = conn.cursor()
        rng = np.random.default_rng(777)
        vec = rng.random(TOKEN_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        vec_str = ",".join(str(float(v)) for v in vec)
        try:
            cur.execute(
                f"INSERT INTO {test_table} (doc_id, text_content, embedding) "
                f"VALUES (?, ?, TO_VECTOR(?, FLOAT, {TOKEN_DIM}))",
                ["doc_new_0001", "Newly inserted document.", vec_str],
            )
            conn.commit()
        finally:
            cur.close()
        results = engine.execute_cypher(
            "MATCH (d:NewRowLabel) WHERE d.doc_id = 'doc_new_0001' RETURN d.doc_id"
        )
        assert len(results) > 0


# --- US2: Custom IRIS table ---


@pytest.mark.integration
class TestAttachCustomTable:
    def test_custom_schema_table(self, conn, engine):
        p = _make_pipeline(conn, engine)
        table = "Test069.ClinicalNotes"
        _create_test_table(conn, table, 50, dim=384)
        try:
            result = p.attach_existing_corpus(
                source_table=table,
                id_col="doc_id",
                text_col="text_content",
                embedding_col="embedding",
                graph_label="ClinNote",
            )
            assert result["label"] == "ClinNote"
            assert result["row_count"] == 50
            results = engine.execute_cypher(
                "MATCH (n:ClinNote) RETURN n.doc_id LIMIT 3"
            )
            assert len(results) > 0
        finally:
            _drop_test_table(conn, table)


# --- US3: Dimension mismatch ---


@pytest.mark.integration
class TestDimensionMismatch:
    def test_mismatch_raises_error(self, pipeline, test_table):
        from iris_vector_rag.pipelines.hybrid_graphrag import DimensionMismatchError

        pipeline.attach_existing_corpus(
            source_table=test_table,
            id_col="doc_id",
            text_col="text_content",
            embedding_col="embedding",
            graph_label="DimTest",
        )
        wrong_dim_vec = np.random.default_rng(0).random(768).tolist()
        with pytest.raises(DimensionMismatchError):
            pipeline._validate_query_dimension("DimTest", wrong_dim_vec)

    def test_all_null_embeddings_warns(self, conn, engine, caplog):
        p = _make_pipeline(conn, engine)
        table = "Test069.NullEmb"
        cur = conn.cursor()
        try:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()
            cur.execute(
                f"CREATE TABLE {table} ("
                f"doc_id VARCHAR(50) PRIMARY KEY, "
                f"text_content VARCHAR(200), "
                f"embedding VECTOR(DOUBLE, 384))"
            )
            conn.commit()
            cur.execute(
                f"INSERT INTO {table} (doc_id, text_content) VALUES ('n1', 'No embedding')"
            )
            conn.commit()
        finally:
            cur.close()
        try:
            with caplog.at_level(logging.WARNING):
                result = p.attach_existing_corpus(
                    source_table=table,
                    id_col="doc_id",
                    text_col="text_content",
                    embedding_col="embedding",
                    graph_label="NullLabel",
                )
            assert result["dimension"] is None or result["dimension"] == 0
            assert any(
                "non-NULL" in r.message or "NULL" in r.message for r in caplog.records
            )
        finally:
            _drop_test_table(conn, table)


# --- Polish: Edge cases ---


@pytest.mark.integration
class TestEdgeCases:
    def test_nonexistent_table_raises(self, pipeline):
        with pytest.raises(ValueError, match="not found"):
            pipeline.attach_existing_corpus(
                source_table="NoSuch.Table",
                id_col="id",
                text_col="txt",
                embedding_col="emb",
                graph_label="Ghost",
            )

    def test_nonexistent_column_raises(self, pipeline, test_table):
        with pytest.raises(ValueError, match="not found"):
            pipeline.attach_existing_corpus(
                source_table=test_table,
                id_col="doc_id",
                text_col="text_content",
                embedding_col="no_such_column",
                graph_label="BadCol",
            )

    def test_upsert_repoint_label(self, conn, engine):
        p = _make_pipeline(conn, engine)
        table_a = "Test069.TableA"
        table_b = "Test069.TableB"
        _create_test_table(conn, table_a, 10)
        _create_test_table(conn, table_b, 20)
        try:
            r1 = p.attach_existing_corpus(
                source_table=table_a,
                id_col="doc_id",
                text_col="text_content",
                embedding_col="embedding",
                graph_label="Repoint",
            )
            assert r1["row_count"] == 10
            r2 = p.attach_existing_corpus(
                source_table=table_b,
                id_col="doc_id",
                text_col="text_content",
                embedding_col="embedding",
                graph_label="Repoint",
            )
            assert r2["row_count"] == 20
            assert r2["table"] == table_b
        finally:
            _drop_test_table(conn, table_a)
            _drop_test_table(conn, table_b)
