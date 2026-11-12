"""
Contract tests for the vector search's diagnostic logging (LOG-004).

Contract: LOG-004 (specs/033-fix-graphrag-vector/contracts/diagnostic_logging_contract.md)
Requirements: FR-004

The diagnostics are the store's (`iris_vector_rag.storage.vector_store_iris`), so each
assertion names that logger: iris_vector_graph logs a "Vector search returned 0 results"
of its own, which must not satisfy a test about the store. Two paths, both deterministic:

- a HybridGraphRAG query on the known corpus (the graph is empty, so it answers from the
  store's vector search, and the query diagnostics are logged);
- a store search under a metadata filter nothing matches, which returns no rows and so
  logs the zero-result diagnostics (row counts, empty scores). The row counts are logged
  only to explain an empty result.
"""

import logging
import re
from io import StringIO

import pytest

STORE = "iris_vector_rag.storage.vector_store_iris"
KNOWN_QUERY = "Which salamander can regrow its legs?"


def capture(level=logging.DEBUG):
    """A handler on the root logger, at `level`, and the stream it writes to."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s - %(message)s"))
    return stream, handler


@pytest.fixture
def store_log():
    """Every record at DEBUG and above, formatted as `LEVEL logger - message`."""
    stream, handler = capture()
    root = logging.getLogger()
    previous = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    yield stream
    root.removeHandler(handler)
    root.setLevel(previous)


def store_lines(stream, level=None):
    """The store's log lines, optionally only those at `level`."""
    lines = [l for l in stream.getvalue().splitlines() if f" {STORE} - " in l]
    if level:
        lines = [l for l in lines if l.startswith(level + " ")]
    return [l.split(f" {STORE} - ", 1)[1] for l in lines]


def query_the_known_corpus(pipeline, top_k=3):
    result = pipeline.query(KNOWN_QUERY, top_k=top_k, generate_answer=False)
    assert result["metadata"]["retrieval_method"] == "vector_fallback"
    assert result["retrieved_documents"][0].id == "kc-axolotl"
    return result


def search_that_finds_nothing(pipeline):
    """The store's vector search under a filter no row matches."""
    results = pipeline.vector_store.similarity_search_by_embedding(
        [0.1] * 384, top_k=3, filter={"source": "kc-no-such-source"}
    )
    assert results == []


class TestDiagnosticLoggingContract:
    """Contract tests for diagnostic logging (LOG-004)."""

    def test_logs_zero_results_message(self, hybrid_on_known_corpus, store_log):
        """FR-004: an empty vector search logs "Vector search returned 0 results" at INFO."""
        search_that_finds_nothing(hybrid_on_known_corpus)

        assert "Vector search returned 0 results" in store_lines(store_log, "INFO")

    def test_logs_query_embedding_dimensions(self, hybrid_on_known_corpus, store_log):
        """FR-004: the query embedding's width is logged: 384 for all-MiniLM-L6-v2."""
        query_the_known_corpus(hybrid_on_known_corpus)

        assert "Query embedding dimensions: 384" in store_lines(store_log, "DEBUG")

    def test_logs_total_documents(self, hybrid_on_known_corpus, store_log):
        """FR-004: an empty result logs the table's row count, the known corpus included."""
        search_that_finds_nothing(hybrid_on_known_corpus)

        counts = [
            int(m.group(1))
            for line in store_lines(store_log, "DEBUG")
            if (
                m := re.fullmatch(
                    r"Total documents in RAG\.SourceDocuments: (\d+)", line
                )
            )
        ]
        assert counts and counts[0] >= 3, store_lines(store_log)

    def test_logs_documents_with_embeddings(self, hybrid_on_known_corpus, store_log):
        """FR-004: an empty result logs how many rows have embeddings (the corpus's 3 do)."""
        search_that_finds_nothing(hybrid_on_known_corpus)

        counts = [
            int(m.group(1))
            for line in store_lines(store_log, "DEBUG")
            if (m := re.fullmatch(r"Documents with embeddings: (\d+)", line))
        ]
        assert counts and counts[0] >= 3, store_lines(store_log)

    def test_logs_sql_query_executed(self, hybrid_on_known_corpus, store_log):
        """FR-004: the SQL the search ran is logged, with its vector function and TOP."""
        query_the_known_corpus(hybrid_on_known_corpus, top_k=3)

        sql = [
            l
            for l in store_lines(store_log, "DEBUG")
            if l.startswith("SQL query executed: ")
        ]
        assert sql, store_lines(store_log)
        assert "SELECT TOP 3 " in sql[0] and "VECTOR_DOT_PRODUCT(" in sql[0], sql[0]

    def test_logs_top_k_parameter(self, hybrid_on_known_corpus, store_log):
        """FR-004: the top-K the search ran with is logged, as asked."""
        query_the_known_corpus(hybrid_on_known_corpus, top_k=2)

        assert "Top-K parameter: 2" in store_lines(store_log, "DEBUG")

    def test_logs_similarity_scores_when_zero_results(
        self, hybrid_on_known_corpus, store_log
    ):
        """FR-004: an empty result logs its scores as an empty list."""
        search_that_finds_nothing(hybrid_on_known_corpus)

        assert "Similarity scores: []" in store_lines(store_log, "DEBUG")

    def test_logging_level_info_shows_high_level_status(self, hybrid_on_known_corpus):
        """FR-004: at INFO the empty result is reported, and none of the DEBUG detail."""
        stream, handler = capture(logging.INFO)
        root = logging.getLogger()
        previous = root.level
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            search_that_finds_nothing(hybrid_on_known_corpus)
        finally:
            root.removeHandler(handler)
            root.setLevel(previous)

        lines = store_lines(stream)
        assert "Vector search returned 0 results" in lines, lines
        assert not any(
            l.startswith(("SQL query executed", "Query embedding dimensions"))
            for l in lines
        ), lines

    def test_logging_level_debug_shows_detailed_diagnostics(
        self, hybrid_on_known_corpus, store_log
    ):
        """FR-004: at DEBUG every query diagnostic is logged, and the row counts too
        when the result is empty. All of them, not most: the old "at least 5 of 6"
        threshold hid a missing one."""
        query_the_known_corpus(hybrid_on_known_corpus, top_k=3)
        search_that_finds_nothing(hybrid_on_known_corpus)

        lines = store_lines(store_log, "DEBUG")
        expected = [
            r"Query embedding dimensions: 384",
            r"Top-K parameter: 3",
            r"SQL query executed: SELECT TOP 3 .*VECTOR_DOT_PRODUCT\(.*",
            r"Vector search returned 3 results",
            r"Similarity scores: \[.+\]",
            r"Similarity scores: \[\]",
            r"Total documents in RAG\.SourceDocuments: \d+",
            r"Documents with embeddings: \d+",
        ]
        missing = [p for p in expected if not any(re.fullmatch(p, l) for l in lines)]
        assert not missing, f"missing {missing} in {lines}"
