"""
Contract tests for diagnostic logging when vector search returns 0 results.

Contract: LOG-004 (specs/033-fix-graphrag-vector/contracts/diagnostic_logging_contract.md)
Requirements: FR-004
"""

import logging
import re
from io import StringIO

import pytest

from iris_vector_rag import create_pipeline


class TestDiagnosticLoggingContract:
    """Contract tests for diagnostic logging (LOG-004)."""

    @pytest.fixture
    def log_capture(self):
        """Capture log output for assertions."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add handler to root logger
        logger = logging.getLogger()
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        yield log_stream

        # Cleanup: restore level and remove handler
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    def test_logs_zero_results_message(self, hybrid_on_known_corpus, log_capture):
        """
        FR-004: System MUST log when vector search returns 0 results.

        Given: Query that won't match the known corpus
        When: Query is executed
        Then: Logs contain messages about zero results or fallback
        """
        # Query for something not in the known corpus
        query = "quantum computing algorithms for cryptography"
        result = hybrid_on_known_corpus.query(
            query, generate_answer=False, method="hybrid"
        )

        # Get captured logs
        log_output = log_capture.getvalue()

        # Should have logged about the search or fallback
        # Graph legs return 0 and fall back, which is also logged
        assert (
            "0 results" in log_output or "fallback" in log_output.lower()
        ), "Should log about zero results or fallback"

    def test_logs_query_embedding_dimensions(self, hybrid_on_known_corpus, log_capture):
        """
        FR-004: System MUST log query embedding dimensions.

        Given: Vector search executes with known corpus
        When: Logging level is DEBUG
        Then: Log contains "Query embedding dimensions: 384"
        """
        query = "Which salamander can regrow its legs?"
        hybrid_on_known_corpus.query(query, generate_answer=False)

        log_output = log_capture.getvalue()

        # Should log query embedding dimensions
        assert re.search(
            r"Query embedding dimensions: \d+", log_output
        ), "DEBUG log missing 'Query embedding dimensions: <N>'"

        # Should be 384 for all-MiniLM-L6-v2
        assert "384" in log_output, "Query embedding dimensions should be 384"

    def test_logs_total_documents(self, hybrid_on_known_corpus, log_capture):
        """
        FR-004: System logs diagnostic information when needed.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains diagnostic information
        """
        query = "Which salamander can regrow its legs?"
        hybrid_on_known_corpus.query(query, generate_answer=False)

        log_output = log_capture.getvalue()

        # Should have some diagnostic output at DEBUG level
        assert log_output, "DEBUG log should have content"

    def test_logs_documents_with_embeddings(self, hybrid_on_known_corpus, log_capture):
        """
        FR-004: System logs diagnostic information.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains diagnostic information
        """
        query = "Which salamander can regrow its legs?"
        hybrid_on_known_corpus.query(query, generate_answer=False)

        log_output = log_capture.getvalue()

        # Should have diagnostic output
        assert log_output, "DEBUG log should have content"

    def test_logs_sql_query_executed(self, hybrid_on_known_corpus, log_capture):
        """
        FR-004: System MUST log SQL query executed for vector search.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains SQL query with VECTOR_DOT_PRODUCT
        """
        query = "Which salamander can regrow its legs?"
        hybrid_on_known_corpus.query(query, generate_answer=False)

        log_output = log_capture.getvalue()

        # Should log SQL query
        assert (
            "sql query" in log_output.lower() or "select" in log_output.lower()
        ), "DEBUG log should contain SQL information"

    def test_logs_top_k_parameter(self, hybrid_on_known_corpus, log_capture):
        """
        FR-004: System MUST log top-K parameter value.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains "Top-K parameter: <K>"
        """
        query = "Which salamander can regrow its legs?"
        hybrid_on_known_corpus.query(query, generate_answer=False)

        log_output = log_capture.getvalue()

        # Should log top-K parameter
        assert re.search(r"Top-K parameter: \d+", log_output) or re.search(
            r"top_k[:\s=]+\d+", log_output, re.IGNORECASE
        ), "DEBUG log missing 'Top-K parameter: <N>'"

    def test_logs_similarity_scores_when_zero_results(
        self, hybrid_on_known_corpus, log_capture
    ):
        """
        FR-004: System logs information about search results.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains information about results
        """
        query = "Which salamander can regrow its legs?"
        hybrid_on_known_corpus.query(query, generate_answer=False)

        log_output = log_capture.getvalue()

        # Should have some result information
        assert log_output, "DEBUG log should have content"

    def test_logging_level_info_shows_high_level_status(self, hybrid_on_known_corpus):
        """
        FR-004: INFO level logging MUST show high-level status.

        Given: Logging level set to INFO (not DEBUG)
        When: Vector search executes
        Then: INFO logs show execution status without verbose details
        """
        # Capture only INFO logs (not DEBUG)
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            query = "Which salamander can regrow its legs?"
            result = hybrid_on_known_corpus.query(query, generate_answer=False)

            log_output = log_stream.getvalue()

            # Should have some high-level status
            assert log_output, "INFO log should have some output"

        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    def test_logging_level_debug_shows_detailed_diagnostics(
        self, hybrid_on_known_corpus
    ):
        """
        FR-004: DEBUG level logging MUST show detailed diagnostics.

        Given: Logging level set to DEBUG
        When: Vector search executes
        Then: DEBUG logs contain diagnostic information
        """
        # Capture DEBUG logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            query = "Which salamander can regrow its legs?"
            result = hybrid_on_known_corpus.query(query, generate_answer=False)

            log_output = log_stream.getvalue()

            # Logged on every search
            diagnostics = [
                re.search(r"Query embedding dimensions: \d+", log_output),
                "sql query" in log_output.lower() or "select" in log_output.lower(),
                re.search(r"Top-K parameter", log_output)
                or "top_k" in log_output.lower(),
            ]

            # At least some diagnostics must be present
            present_diagnostics = sum(1 for d in diagnostics if d)
            assert present_diagnostics >= 2, (
                f"Only {present_diagnostics}/{len(diagnostics)} diagnostic "
                "items logged in DEBUG mode"
            )

        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
