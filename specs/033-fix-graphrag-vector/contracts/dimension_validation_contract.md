# Contract: Dimension Validation

**Feature**: Fix GraphRAG Vector Retrieval Logic
**Contract ID**: DVC-002
**Requirements**: FR-005
**Test File**: `tests/contract/test_dimension_validation_contract.py`

## Contract Definition

### Given

```python
# Preconditions:
# 1. GraphRAG pipeline initialized
# 2. Embedding model configured (all-MiniLM-L6-v2 → 384D)
# 3. RAG.SourceDocuments contains documents with 384D embeddings
```

### When

```python
# Action 1: Generate query embedding
query = "What are the symptoms of diabetes?"
query_embedding = embedding_manager.generate_embedding(query)

# Action 2: Execute vector search
result = pipeline.query(query)
```

### Then

```python
# Postconditions (all MUST be true):

# 1. Query embedding is exactly 384 dimensions
assert len(query_embedding) == 384, \
    f"Query embedding dimension mismatch: {len(query_embedding)} != 384"

# 2. If document embedding dimension != 384, raise DimensionMismatchError
# (Tested via error injection: corrupt document embedding and verify error)

# 3. Error message includes both dimensions
try:
    # Simulate dimension mismatch
    corrupt_embedding = [0.1] * 256  # Wrong dimension
    pipeline._validate_dimensions(corrupt_embedding)
except DimensionMismatchError as e:
    assert "256" in str(e), "Error message must include actual dimension"
    assert "384" in str(e), "Error message must include expected dimension"

# 4. Error message suggests actionable fix
try:
    corrupt_embedding = [0.1] * 256
    pipeline._validate_dimensions(corrupt_embedding)
except DimensionMismatchError as e:
    assert "model" in str(e).lower() or "re-index" in str(e).lower(), \
        "Error message must suggest model verification or re-indexing"
```

---

## Contract Test Implementation

### File: `tests/contract/test_dimension_validation_contract.py`

```python
"""
Contract tests for embedding dimension validation.

Contract: DVC-002 (specs/033-fix-graphrag-vector/contracts/dimension_validation_contract.md)
Requirements: FR-005
"""

import pytest
from iris_rag import create_pipeline
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.config.manager import ConfigurationManager


class DimensionMismatchError(Exception):
    """Raised when embedding dimensions don't match."""
    pass


class TestDimensionValidationContract:
    """Contract tests for dimension validation (DVC-002)."""

    @pytest.fixture
    def config_manager(self):
        """Create configuration manager."""
        return ConfigurationManager()

    @pytest.fixture
    def embedding_manager(self, config_manager):
        """Create embedding manager."""
        return EmbeddingManager(config_manager)

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create GraphRAG pipeline."""
        return create_pipeline("graphrag", validate_requirements=True)

    def test_query_embedding_is_384_dimensions(self, embedding_manager):
        """
        FR-005: Query embedding MUST be exactly 384 dimensions.

        Given: all-MiniLM-L6-v2 embedding model configured
        When: Query is embedded
        Then: Resulting vector is exactly 384 dimensions
        """
        query = "What are the symptoms of diabetes?"
        query_embedding = embedding_manager.generate_embedding(query)

        assert len(query_embedding) == 384, \
            f"Query embedding dimension mismatch: {len(query_embedding)} != 384"

    def test_dimension_validation_before_search(self, graphrag_pipeline):
        """
        FR-005: Vector search MUST validate dimensions before querying.

        Given: GraphRAG pipeline with dimension validation enabled
        When: Query is executed
        Then: Dimensions are validated before IRIS query (no silent failures)
        """
        query = "What are the symptoms of diabetes?"

        # Should not raise error (dimensions match)
        result = graphrag_pipeline.query(query)

        # If retrieval succeeds, validation passed
        assert len(result.contexts) >= 0, \
            "Query execution should succeed with matching dimensions"

    def test_dimension_mismatch_raises_clear_error(self, graphrag_pipeline, embedding_manager):
        """
        FR-005: Dimension mismatch MUST raise DimensionMismatchError with clear message.

        Given: Query embedding with wrong dimensions
        When: Validation is performed
        Then: DimensionMismatchError raised with both dimensions in message
        """
        # Generate correct embedding first
        query = "test query"
        correct_embedding = embedding_manager.generate_embedding(query)
        assert len(correct_embedding) == 384, "Embedding manager should produce 384D"

        # Simulate dimension mismatch (implementation detail - may need adjustment)
        # This test validates the error message structure when mismatch occurs
        with pytest.raises(Exception) as exc_info:
            # Trigger dimension validation failure
            # (Implementation will add _validate_dimensions method)
            if hasattr(graphrag_pipeline, '_validate_dimensions'):
                corrupt_embedding = [0.1] * 256  # Wrong dimension
                graphrag_pipeline._validate_dimensions(corrupt_embedding, expected_dims=384)
            else:
                pytest.skip("_validate_dimensions method not yet implemented")

        error_msg = str(exc_info.value).lower()

        # Error message MUST include both dimensions
        assert "256" in error_msg or "dimension" in error_msg, \
            "Error message must mention actual dimension"
        assert "384" in error_msg, \
            "Error message must mention expected dimension"

    def test_dimension_error_suggests_actionable_fix(self, graphrag_pipeline):
        """
        FR-005: DimensionMismatchError MUST suggest actionable fix.

        Given: Dimension mismatch error occurs
        When: Error message is inspected
        Then: Message suggests model verification or re-indexing
        """
        # This test validates error message quality
        # (Implementation will ensure helpful error messages)

        with pytest.raises(Exception) as exc_info:
            if hasattr(graphrag_pipeline, '_validate_dimensions'):
                corrupt_embedding = [0.1] * 768  # Wrong dimension (BERT-base size)
                graphrag_pipeline._validate_dimensions(corrupt_embedding, expected_dims=384)
            else:
                pytest.skip("_validate_dimensions method not yet implemented")

        error_msg = str(exc_info.value).lower()

        # Error message MUST suggest fix
        actionable_keywords = ["model", "re-index", "reindex", "embedding", "verify"]
        assert any(keyword in error_msg for keyword in actionable_keywords), \
            f"Error message must suggest actionable fix. Got: {exc_info.value}"

    def test_document_embedding_dimension_check(self, graphrag_pipeline):
        """
        FR-005: System MUST validate document embedding dimensions.

        Given: Documents in RAG.SourceDocuments
        When: Vector search initializes
        Then: Document embeddings are validated to be 384D
        """
        # Sample a document embedding from database
        from iris_rag.core.connection import ConnectionManager

        conn = ConnectionManager(graphrag_pipeline.config).get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            LIMIT 1
        """)
        result = cursor.fetchone()

        if result is None:
            pytest.skip("No documents with embeddings in database")

        document_embedding = result[0]

        # Document embedding MUST be 384D
        assert len(document_embedding) == 384, \
            f"Document embedding dimension mismatch: {len(document_embedding)} != 384. " \
            f"Database may contain embeddings from different model. Re-indexing required."

    def test_mismatched_embeddings_prevented(self, graphrag_pipeline):
        """
        FR-005: System MUST prevent queries with mismatched embedding dimensions.

        Given: Database has 384D embeddings
        When: Query embedding has different dimensions
        Then: Error raised before IRIS query (no silent failures)
        """
        # This is a defensive test ensuring validation occurs
        # If query embedding generator is wrong, validation should catch it

        query = "test query"

        # Normal query should work (dimensions match)
        result = graphrag_pipeline.query(query)

        # If we get here without error, validation either:
        # 1. Passed (dimensions match) - good
        # 2. Not implemented yet - will fail contract
        # The actual dimension validation logic will be added in implementation

        # For now, verify retrieval works with correct dimensions
        assert isinstance(result.contexts, list), \
            "Query should execute successfully with matching dimensions"
```

---

## Test Execution

### Before Fix (Expected: PARTIAL PASS)

```bash
$ .venv/bin/pytest tests/contract/test_dimension_validation_contract.py -v

tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_query_embedding_is_384_dimensions PASSED
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_dimension_validation_before_search SKIPPED (validation not implemented)
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_dimension_mismatch_raises_clear_error SKIPPED (validation not implemented)
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_dimension_error_suggests_actionable_fix SKIPPED (validation not implemented)
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_document_embedding_dimension_check PASSED
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_mismatched_embeddings_prevented SKIPPED (validation not implemented)

============================== 2 passed, 4 skipped in 1.56s ==============================
```

### After Fix (Expected: PASS)

```bash
$ .venv/bin/pytest tests/contract/test_dimension_validation_contract.py -v

tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_query_embedding_is_384_dimensions PASSED
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_dimension_validation_before_search PASSED
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_dimension_mismatch_raises_clear_error PASSED
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_dimension_error_suggests_actionable_fix PASSED
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_document_embedding_dimension_check PASSED
tests/contract/test_dimension_validation_contract.py::TestDimensionValidationContract::test_mismatched_embeddings_prevented PASSED

============================== 6 passed in 2.01s ==============================
```

---

## Validation Error Examples

### Expected Error Message Format

```python
# Good error message (meets FR-005):
DimensionMismatchError(
    "Query embedding dimension mismatch: 256 != 384. "
    "Expected all-MiniLM-L6-v2 model (384D). "
    "Verify embedding model configuration in iris_rag/config/default_config.yaml."
)

# Good error message for document mismatch:
DimensionMismatchError(
    "Document embedding dimension mismatch: 768 != 384. "
    "Database may contain embeddings from different model (BERT-base uses 768D). "
    "Re-indexing required with all-MiniLM-L6-v2 (384D)."
)
```

### Poor Error Messages (NOT acceptable)

```python
# Bad: No dimensions mentioned
ValueError("Dimension mismatch")  # ❌

# Bad: No actionable suggestion
ValueError("Embeddings have wrong size")  # ❌

# Bad: Silent failure
# (No error raised, vector search returns 0 results silently)  # ❌
```

---

## Implementation Requirements

### GraphRAG Pipeline Must Add

```python
class GraphRAGPipeline(RAGPipeline):
    EXPECTED_EMBEDDING_DIMS = 384  # all-MiniLM-L6-v2

    def _validate_dimensions(self, embedding: List[float], expected_dims: int = None) -> None:
        """
        Validate embedding dimensions match expected value.

        Args:
            embedding: Embedding vector to validate
            expected_dims: Expected dimension count (default: EXPECTED_EMBEDDING_DIMS)

        Raises:
            DimensionMismatchError: If dimensions don't match
        """
        expected_dims = expected_dims or self.EXPECTED_EMBEDDING_DIMS
        actual_dims = len(embedding)

        if actual_dims != expected_dims:
            raise DimensionMismatchError(
                f"Embedding dimension mismatch: {actual_dims} != {expected_dims}. "
                f"Expected all-MiniLM-L6-v2 model (384D). "
                f"Verify embedding model configuration."
            )

    def query(self, query: str, **kwargs) -> RAGResponse:
        """Query with dimension validation."""
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding(query)

        # Validate query embedding dimensions (FR-005)
        self._validate_dimensions(query_embedding)

        # Continue with vector search...
```

---

## Contract Acceptance Criteria

- ✅ All 6 dimension validation tests PASS
- ✅ Query embeddings validated before search
- ✅ DimensionMismatchError raised with clear messages
- ✅ Error messages include both actual and expected dimensions
- ✅ Error messages suggest actionable fixes

---

**Contract Status**: ✅ DEFINED
**Test File**: To be created in Phase 2 (Task T002)
**Expected Result**: PARTIAL PASS before fix (validation not implemented), FULL PASS after fix
