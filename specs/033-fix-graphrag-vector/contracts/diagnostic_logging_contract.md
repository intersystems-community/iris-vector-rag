# Contract: Diagnostic Logging

**Feature**: Fix GraphRAG Vector Retrieval Logic
**Contract ID**: LOG-004
**Requirements**: FR-004
**Test File**: `tests/contract/test_diagnostic_logging_contract.py`

## Contract Definition

### Given

```python
# Preconditions:
# 1. GraphRAG pipeline initialized with logging enabled
# 2. Logging level set to INFO or DEBUG
# 3. Vector search executes and returns 0 results (or >0 results)
```

### When

```python
# Action: Execute query with logging enabled
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = create_pipeline("graphrag")
result = pipeline.query("What are the symptoms of diabetes?")
```

### Then (when 0 results returned)

```python
# Postconditions (all MUST be logged):

# 1. INFO log: "Vector search returned 0 results"
assert "Vector search returned 0 results" in captured_logs

# 2. DEBUG log: Query embedding dimensions
assert re.search(r"Query embedding dimensions: \d+", captured_logs)
assert "384" in captured_logs  # Expected dimension

# 3. DEBUG log: Total documents in RAG.SourceDocuments
assert re.search(r"Total documents in RAG\.SourceDocuments: \d+", captured_logs)

# 4. DEBUG log: Documents with embeddings
assert re.search(r"Documents with embeddings: \d+", captured_logs)

# 5. DEBUG log: SQL query executed
assert "SQL query executed:" in captured_logs or "SQL query:" in captured_logs
assert "VECTOR_DOT_PRODUCT" in captured_logs

# 6. DEBUG log: Top-K parameter
assert re.search(r"Top-K parameter: \d+", captured_logs)

# 7. DEBUG log: Sample similarity scores (when 0 results)
assert "Sample similarity scores:" in captured_logs or "similarity scores" in captured_logs.lower()
```

---

## Contract Test Implementation

### File: `tests/contract/test_diagnostic_logging_contract.py`

```python
"""
Contract tests for diagnostic logging when vector search returns 0 results.

Contract: LOG-004 (specs/033-fix-graphrag-vector/contracts/diagnostic_logging_contract.md)
Requirements: FR-004
"""

import logging
import re
from io import StringIO
import pytest
from iris_rag import create_pipeline


class TestDiagnosticLoggingContract:
    """Contract tests for diagnostic logging (LOG-004)."""

    @pytest.fixture
    def log_capture(self):
        """Capture log output for assertions."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add handler to root logger
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        yield log_stream

        # Cleanup
        logger.removeHandler(handler)

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create GraphRAG pipeline."""
        return create_pipeline("graphrag", validate_requirements=True)

    def test_logs_zero_results_message(self, graphrag_pipeline, log_capture):
        """
        FR-004: System MUST log when vector search returns 0 results.

        Given: Vector search executes
        When: 0 results are returned
        Then: INFO log contains "Vector search returned 0 results"
        """
        # Execute query (may or may not return 0 results)
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        # Get captured logs
        log_output = log_capture.getvalue()

        # If 0 results, message MUST be logged
        if len(result.contexts) == 0:
            assert "Vector search returned 0 results" in log_output, \
                "INFO log 'Vector search returned 0 results' missing when 0 results returned"

    def test_logs_query_embedding_dimensions(self, graphrag_pipeline, log_capture):
        """
        FR-004: System MUST log query embedding dimensions.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains "Query embedding dimensions: 384"
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        log_output = log_capture.getvalue()

        # Should log query embedding dimensions
        assert re.search(r"Query embedding dimensions: \d+", log_output), \
            "DEBUG log missing 'Query embedding dimensions: <N>'"

        # Should be 384 for all-MiniLM-L6-v2
        assert "384" in log_output, \
            "Query embedding dimensions should be 384"

    def test_logs_total_documents(self, graphrag_pipeline, log_capture):
        """
        FR-004: System MUST log total documents in RAG.SourceDocuments.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains "Total documents in RAG.SourceDocuments: <count>"
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        log_output = log_capture.getvalue()

        # Should log total documents
        assert re.search(r"Total documents in RAG\.SourceDocuments: \d+", log_output) or \
               re.search(r"Total documents: \d+", log_output), \
            "DEBUG log missing 'Total documents in RAG.SourceDocuments: <N>'"

    def test_logs_documents_with_embeddings(self, graphrag_pipeline, log_capture):
        """
        FR-004: System MUST log count of documents with embeddings.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains "Documents with embeddings: <count>"
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        log_output = log_capture.getvalue()

        # Should log documents with embeddings count
        assert re.search(r"Documents with embeddings: \d+", log_output), \
            "DEBUG log missing 'Documents with embeddings: <N>'"

    def test_logs_sql_query_executed(self, graphrag_pipeline, log_capture):
        """
        FR-004: System MUST log SQL query executed for vector search.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains SQL query with VECTOR_DOT_PRODUCT
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        log_output = log_capture.getvalue()

        # Should log SQL query
        assert "SQL query" in log_output.lower(), \
            "DEBUG log missing 'SQL query executed:' or 'SQL query:'"

        # SQL should contain VECTOR_DOT_PRODUCT (IRIS vector search function)
        assert "VECTOR_DOT_PRODUCT" in log_output, \
            "SQL query should use VECTOR_DOT_PRODUCT for vector search"

    def test_logs_top_k_parameter(self, graphrag_pipeline, log_capture):
        """
        FR-004: System MUST log top-K parameter value.

        Given: Vector search executes
        When: Logging level is DEBUG
        Then: Log contains "Top-K parameter: <K>"
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        log_output = log_capture.getvalue()

        # Should log top-K parameter
        assert re.search(r"Top-K parameter: \d+", log_output) or \
               re.search(r"top_k[:\s=]+\d+", log_output, re.IGNORECASE), \
            "DEBUG log missing 'Top-K parameter: <N>'"

    def test_logs_similarity_scores_when_zero_results(self, graphrag_pipeline, log_capture):
        """
        FR-004: System MUST log similarity scores (or lack thereof) when 0 results.

        Given: Vector search returns 0 results
        When: Logging level is DEBUG
        Then: Log contains "Sample similarity scores: None returned" or similar
        """
        query = "What are the symptoms of diabetes?"
        result = graphrag_pipeline.query(query)

        log_output = log_capture.getvalue()

        # Only check if 0 results returned
        if len(result.contexts) == 0:
            # Should log something about similarity scores
            assert "similarity scores" in log_output.lower() or \
                   "scores" in log_output.lower(), \
                "DEBUG log missing similarity scores information when 0 results"

    def test_logging_level_info_shows_high_level_status(self, graphrag_pipeline):
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
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            query = "What are the symptoms of diabetes?"
            result = graphrag_pipeline.query(query)

            log_output = log_stream.getvalue()

            # INFO should have high-level status
            if len(result.contexts) == 0:
                assert "Vector search returned 0 results" in log_output, \
                    "INFO log should show 0 results status"
            else:
                # Should show successful retrieval
                assert "Vector search returned" in log_output or \
                       "retrieved" in log_output.lower(), \
                    "INFO log should show retrieval status"

        finally:
            logger.removeHandler(handler)

    def test_logging_level_debug_shows_detailed_diagnostics(self, graphrag_pipeline):
        """
        FR-004: DEBUG level logging MUST show detailed diagnostics.

        Given: Logging level set to DEBUG
        When: Vector search executes
        Then: DEBUG logs contain all diagnostic information
        """
        # Capture DEBUG logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            query = "What are the symptoms of diabetes?"
            result = graphrag_pipeline.query(query)

            log_output = log_stream.getvalue()

            # DEBUG should have detailed diagnostics (at least 5 of the following)
            diagnostics = [
                re.search(r"Query embedding dimensions: \d+", log_output),
                re.search(r"Total documents", log_output),
                re.search(r"Documents with embeddings", log_output),
                "SQL query" in log_output.lower(),
                re.search(r"Top-K parameter", log_output),
                "VECTOR_DOT_PRODUCT" in log_output,
            ]

            # At least 5 diagnostic items should be present
            present_diagnostics = sum(1 for d in diagnostics if d)
            assert present_diagnostics >= 5, \
                f"Only {present_diagnostics}/6 diagnostic items logged in DEBUG mode"

        finally:
            logger.removeHandler(handler)
```

---

## Test Execution

### Before Fix (Expected: FAIL/SKIP)

```bash
$ .venv/bin/pytest tests/contract/test_diagnostic_logging_contract.py -v

tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_zero_results_message SKIPPED (not implemented)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_query_embedding_dimensions SKIPPED (not implemented)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_total_documents SKIPPED (not implemented)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_documents_with_embeddings SKIPPED (not implemented)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_sql_query_executed SKIPPED (not implemented)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_top_k_parameter SKIPPED (not implemented)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_similarity_scores_when_zero_results SKIPPED (not implemented)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logging_level_info_shows_high_level_status PASSED (basic logging exists)
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logging_level_debug_shows_detailed_diagnostics SKIPPED (not implemented)

============================== 1 passed, 8 skipped in 1.23s ==============================
```

### After Fix (Expected: PASS)

```bash
$ .venv/bin/pytest tests/contract/test_diagnostic_logging_contract.py -v

tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_zero_results_message PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_query_embedding_dimensions PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_total_documents PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_documents_with_embeddings PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_sql_query_executed PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_top_k_parameter PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logs_similarity_scores_when_zero_results PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logging_level_info_shows_high_level_status PASSED
tests/contract/test_diagnostic_logging_contract.py::TestDiagnosticLoggingContract::test_logging_level_debug_shows_detailed_diagnostics PASSED

============================== 9 passed in 2.45s ==============================
```

---

## Expected Log Output Examples

### Successful Retrieval (>0 results)

```
INFO - Vector search returned 10 results
DEBUG - Query embedding dimensions: 384
DEBUG - Total documents in RAG.SourceDocuments: 2376
DEBUG - Documents with embeddings: 2376
DEBUG - SQL query executed: SELECT TOP 10 id, text_content, VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?, 'DECIMAL')) as score FROM RAG.SourceDocuments WHERE embedding IS NOT NULL ORDER BY score DESC
DEBUG - Top-K parameter: 10
DEBUG - Sample similarity scores: [0.87, 0.82, 0.79, 0.76, 0.73, 0.68, 0.65, 0.61, 0.58, 0.55]
INFO - Query processed successfully | Answer: 250 chars | Contexts: 10
```

### Failed Retrieval (0 results)

```
INFO - Vector search returned 0 results
DEBUG - Query embedding dimensions: 384
DEBUG - Total documents in RAG.SourceDocuments: 2376
DEBUG - Documents with embeddings: 2376
DEBUG - SQL query executed: SELECT TOP 10 id, text_content, VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?, 'DECIMAL')) as score FROM RAG.SourceDocuments WHERE embedding IS NOT NULL ORDER BY score DESC
DEBUG - Top-K parameter: 10
DEBUG - Sample similarity scores: None returned
WARNING - No relevant documents found for query
INFO - Query processed | Answer: 48 chars | Contexts: 0
```

### Dimension Mismatch Error (FR-005 + FR-004)

```
DEBUG - Query embedding dimensions: 256
ERROR - Dimension mismatch detected: 256 != 384
ERROR - Expected all-MiniLM-L6-v2 model (384D)
ERROR - Verify embedding model configuration in iris_rag/config/default_config.yaml
```

---

## Implementation Requirements

### GraphRAG Pipeline Must Add Logging

```python
import logging

class GraphRAGPipeline(RAGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    def query(self, query: str, **kwargs) -> RAGResponse:
        """Query with comprehensive diagnostic logging (FR-004)."""

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(query)
        self.logger.debug(f"Query embedding dimensions: {len(query_embedding)}")

        # Validate dimensions (FR-005)
        self._validate_dimensions(query_embedding)

        # Get database stats
        total_docs, docs_with_embeddings = self._get_document_stats()
        self.logger.debug(f"Total documents in RAG.SourceDocuments: {total_docs}")
        self.logger.debug(f"Documents with embeddings: {docs_with_embeddings}")

        # Perform vector search
        top_k = self.config.get("retrieval.top_k", 10)
        self.logger.debug(f"Top-K parameter: {top_k}")

        sql = self._build_vector_search_sql(top_k)
        self.logger.debug(f"SQL query executed: {sql}")

        retrieved_docs = self._execute_vector_search(query_embedding, top_k)

        # Log results
        if len(retrieved_docs) == 0:
            self.logger.info("Vector search returned 0 results")
            self.logger.debug("Sample similarity scores: None returned")
        else:
            self.logger.info(f"Vector search returned {len(retrieved_docs)} results")
            sample_scores = [doc.similarity_score for doc in retrieved_docs[:5]]
            self.logger.debug(f"Sample similarity scores: {sample_scores}")

        # Continue with answer generation...
```

---

## Contract Acceptance Criteria

- ✅ All 9 diagnostic logging tests PASS
- ✅ INFO logs show high-level status (0 results, N results)
- ✅ DEBUG logs contain all 6 diagnostic items when 0 results
- ✅ Logs are actionable for debugging vector search issues
- ✅ No sensitive data logged (embeddings logged as dimensions only)

---

**Contract Status**: ✅ DEFINED
**Test File**: To be created in Phase 2 (Task T004)
**Expected Result**: PARTIAL PASS before fix (basic logging exists), FULL PASS after fix (comprehensive diagnostics)
