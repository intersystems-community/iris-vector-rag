# Contract: Embedding Dimension Validation (DIM-001)

**Feature**: 036-retrofit-graphrag-s
**Requirements**: FR-021, FR-022, FR-023, FR-024
**Applies To**: BasicRAG, CRAG, BasicRerankRAG, PyLateColBERT

---

## Purpose

Validate that all pipelines verify embedding dimension compatibility between model output and database schema, detecting mismatches before vector operations.

---

## Contract Specification

### 1. Expected Dimensions (Pipeline-Specific)

#### BasicRAG
- **Model**: all-MiniLM-L6-v2 (sentence-transformers)
- **Expected Dimension**: 384
- **Database Schema**: `RAG.SourceDocuments.embedding` → VECTOR(DOUBLE, 384)

#### CRAG
- **Model**: all-MiniLM-L6-v2 (sentence-transformers)
- **Expected Dimension**: 384
- **Database Schema**: Same as BasicRAG

#### BasicRerankRAG
- **Query Model**: all-MiniLM-L6-v2 (384D)
- **Reranker Model**: cross-encoder (variable, typically 768D)
- **Expected Dimension**: 384 for vector search, variable for reranker

#### PyLateColBERT
- **Model**: ColBERT (late interaction)
- **Expected Dimension**: Variable (32 dimensions per token, multiple tokens)
- **Database Schema**: Token embeddings stored separately

---

### 2. Dimension Validation Points (FR-021, FR-022)

**Validation MUST occur**:
1. **Pipeline Initialization**: Validate embedding model configuration
2. **Query Embedding**: Validate query embedding dimensions before search
3. **Document Indexing**: Validate document embedding dimensions before storage
4. **Database Query**: Validate stored embedding dimensions match query

**Validation Logic**:
```python
def validate_dimensions(
    embedding: list[float],
    expected_dim: int,
    context: str
) -> None:
    """
    Validate embedding dimensions match expected.

    Args:
        embedding: Embedding vector to validate
        expected_dim: Expected dimension count
        context: Validation context (e.g., "query", "document")

    Raises:
        DimensionMismatchError: If dimensions don't match
    """
    actual_dim = len(embedding)

    if actual_dim != expected_dim:
        raise DimensionMismatchError(
            f"Embedding dimension mismatch in {context}",
            context={
                "pipeline": self.__class__.__name__,
                "operation": context,
                "model": self.config.embedding_model
            },
            expected=expected_dim,
            actual=actual_dim,
            fix=(
                f"Expected {expected_dim} dimensions ({self.config.embedding_model}), "
                f"but got {actual_dim} dimensions.\n"
                f"Verify embedding model configuration in config.yaml:\n"
                f"  Current: {self.config.embedding_model}\n"
                f"  Expected: all-MiniLM-L6-v2 (384D)\n"
                f"If database has wrong dimensions, re-index documents:\n"
                f"  python scripts/reindex_documents.py"
            )
        )
```

---

### 3. Dimension Mismatch Error Messages (FR-023)

**Error Message Template**:
```
Error: Embedding dimension mismatch in {context}
Context: {pipeline_type}, {operation}, {model_name}
Expected: {expected_dim} dimensions ({expected_model_name})
Actual: {actual_dim} dimensions
Fix: {actionable_steps}
```

**Example Error** (Query Embedding):
```
Error: Embedding dimension mismatch in query
Context: BasicRAG pipeline, vector_search operation, bert-base-uncased
Expected: 384 dimensions (all-MiniLM-L6-v2)
Actual: 768 dimensions
Fix: Expected 384 dimensions (all-MiniLM-L6-v2), but got 768 dimensions.
     Verify embedding model configuration in config.yaml:
       Current: bert-base-uncased (768D)
       Expected: all-MiniLM-L6-v2 (384D)
     Update config.yaml:
       embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
     Then restart pipeline
```

**Example Error** (Document Indexing):
```
Error: Embedding dimension mismatch in document indexing
Context: CRAG pipeline, load_documents operation, all-MiniLM-L6-v2
Expected: 384 dimensions (database schema)
Actual: 768 dimensions (document embeddings)
Fix: Database expects 384D embeddings, but documents have 768D embeddings.
     Database was likely indexed with BERT-base (768D).
     Options:
       1. Re-index database with all-MiniLM-L6-v2:
          python scripts/reindex_documents.py --model all-MiniLM-L6-v2
       2. Update database schema to 768D (BREAKING CHANGE):
          ALTER TABLE RAG.SourceDocuments ALTER COLUMN embedding VECTOR(DOUBLE, 768)
     Recommended: Option 1 (re-index)
```

---

### 4. Actionable Error Guidance (FR-024)

**Required Guidance Elements**:
1. **Problem Diagnosis**: What's wrong (dimension mismatch)
2. **Current State**: Actual dimension and model
3. **Expected State**: Expected dimension and model
4. **Root Cause**: Why mismatch occurred (config change, wrong model, stale database)
5. **Resolution Options**: Numbered list of fixes
6. **Recommended Fix**: Which option to choose

**Guidance Template**:
```
Fix: {diagnosis}
     Current state: {actual_dim}D embeddings ({actual_model})
     Expected state: {expected_dim}D embeddings ({expected_model})
     Root cause: {likely_cause}
     Options:
       1. {option_1_description}
          {option_1_command}
       2. {option_2_description}
          {option_2_command}
     Recommended: {recommended_option}
```

---

### 5. Dimension Transformation Support (FR-024)

**Configuration Option**: Allow dimension transformation when configured

**Config Format** (config.yaml):
```yaml
embedding_config:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  transformation:
    enable: false                    # Enable dimension transformation
    method: "pca"                    # "pca", "truncate", "pad"
    target_dimension: 384            # Target dimension after transformation
```

**Transformation Methods**:
- **PCA**: Reduce dimensions via principal component analysis
- **Truncate**: Truncate to first N dimensions (lossy)
- **Pad**: Pad with zeros to reach target dimension (for smaller → larger)

**Transformation Logic**:
```python
if self.config.embedding_config.transformation.enable:
    if actual_dim > expected_dim:
        # Reduce dimensions (PCA or truncate)
        embedding = reduce_dimensions(
            embedding,
            method=self.config.embedding_config.transformation.method,
            target_dim=expected_dim
        )
    elif actual_dim < expected_dim:
        # Pad dimensions
        embedding = pad_dimensions(embedding, target_dim=expected_dim)
```

---

## Test Implementation

### Test Files
- `tests/contract/test_basic_dimension_validation.py`
- `tests/contract/test_crag_dimension_validation.py`
- `tests/contract/test_basic_rerank_dimension_validation.py`
- `tests/contract/test_pylate_colbert_dimension_validation.py`

### Test Cases

#### Test: Query Embedding Dimension Validation
```python
def test_query_embedding_is_384_dimensions(pipeline):
    """
    FR-021: Query embedding MUST be exactly 384 dimensions.

    Given: all-MiniLM-L6-v2 embedding model configured
    When: Query is embedded
    Then: Resulting vector is exactly 384 dimensions
    """
    query = "What are the symptoms of diabetes?"
    result = pipeline.query(query)

    # Get embedding via pipeline's embedding manager
    embedding = pipeline.embedding_manager.generate_embedding(query)

    assert len(embedding) == 384, \
        f"Query embedding dimension mismatch: {len(embedding)} != 384"
```

#### Test: Dimension Validation Before Search
```python
def test_dimension_validation_before_search(pipeline, mocker):
    """
    FR-022: Vector search MUST validate dimensions before querying.

    Given: Pipeline with dimension validation
    When: Query embedding has wrong dimensions (mocked)
    Then: DimensionMismatchError raised before database query
    """
    query = "test query"

    # Mock embedding to return wrong dimensions
    corrupt_embedding = [0.1] * 768  # Wrong dimension (BERT-base)
    mocker.patch.object(
        pipeline.embedding_manager,
        'generate_embedding',
        return_value=corrupt_embedding
    )

    with pytest.raises(DimensionMismatchError) as exc_info:
        pipeline.query(query)

    error_msg = str(exc_info.value)

    # Error raised BEFORE database query
    assert "768" in error_msg
    assert "384" in error_msg
```

#### Test: Dimension Mismatch Error Includes Both Dimensions
```python
def test_dimension_mismatch_includes_both_dimensions(pipeline, mocker):
    """
    FR-023: DimensionMismatchError MUST include both expected and actual dimensions.

    Given: Dimension validation fails
    When: Error is raised
    Then: Error message includes both 384 (expected) and actual dimension
    """
    # Mock embedding to wrong dimension
    corrupt_embedding = [0.1] * 512
    mocker.patch.object(
        pipeline.embedding_manager,
        'generate_embedding',
        return_value=corrupt_embedding
    )

    with pytest.raises(DimensionMismatchError) as exc_info:
        pipeline.query("test query")

    error_msg = str(exc_info.value)

    # MUST include expected dimension
    assert "384" in error_msg, "Error must include expected dimension (384)"

    # MUST include actual dimension
    assert "512" in error_msg, "Error must include actual dimension (512)"
```

#### Test: Error Suggests Actionable Fix
```python
def test_error_suggests_actionable_fix(pipeline, mocker):
    """
    FR-024: DimensionMismatchError MUST suggest actionable fix.

    Given: Dimension mismatch detected
    When: Error message is inspected
    Then: Message suggests model verification or re-indexing
    """
    # Mock embedding to wrong dimension
    corrupt_embedding = [0.1] * 768
    mocker.patch.object(
        pipeline.embedding_manager,
        'generate_embedding',
        return_value=corrupt_embedding
    )

    with pytest.raises(DimensionMismatchError) as exc_info:
        pipeline.query("test query")

    error_msg = str(exc_info.value).lower()

    # MUST suggest fix
    actionable_keywords = ["verify", "config", "re-index", "model", "update"]
    assert any(keyword in error_msg for keyword in actionable_keywords), \
        f"Error message must suggest actionable fix. Got: {exc_info.value}"
```

#### Test: Document Embedding Dimension Validation
```python
@pytest.mark.requires_database
def test_document_embedding_dimension_check(pipeline):
    """
    FR-021: System MUST validate document embedding dimensions.

    Given: Documents in RAG.SourceDocuments
    When: Document embeddings sampled
    Then: All embeddings are 384D
    """
    from iris_rag.core.connection import ConnectionManager

    conn = ConnectionManager(pipeline.config).get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT embedding FROM RAG.SourceDocuments
        WHERE embedding IS NOT NULL
        LIMIT 5
    """)
    results = cursor.fetchall()

    if not results:
        pytest.skip("No documents with embeddings in database")

    for row in results:
        document_embedding = row[0]
        assert len(document_embedding) == 384, \
            f"Document embedding dimension mismatch: {len(document_embedding)} != 384"
```

#### Test: Dimension Transformation (Optional)
```python
def test_dimension_transformation_when_enabled(pipeline, mocker):
    """
    FR-024: System SHOULD support dimension transformation when configured.

    Given: Dimension transformation enabled in config
    When: Embedding has wrong dimensions
    Then: Dimensions are transformed to match expected
    """
    # Enable transformation
    pipeline.config.embedding_config.transformation.enable = True
    pipeline.config.embedding_config.transformation.method = "truncate"

    # Mock embedding with extra dimensions
    oversized_embedding = [0.1] * 768
    mocker.patch.object(
        pipeline.embedding_manager,
        'generate_embedding',
        return_value=oversized_embedding
    )

    # Should NOT raise error (transformation occurs)
    result = pipeline.query("test query")

    # Verify query succeeded (transformation worked)
    assert result is not None
```

---

## Acceptance Criteria

- ✅ Query embeddings validated before vector search (FR-021)
- ✅ Document embeddings validated during indexing (FR-021)
- ✅ Dimension mismatches detected before database operations (FR-022)
- ✅ Error messages include both expected and actual dimensions (FR-023)
- ✅ Error messages suggest actionable fixes (FR-024)
- ✅ Dimension transformation supported when configured (FR-024)
- ✅ All tests complete in <30 seconds (FR-005)

---

## Notes

- PyLateColBERT dimension validation is more complex (token-level embeddings)
- Dimension transformation is OPTIONAL (pipelines may choose not to support)
- Database schema dimension changes are BREAKING CHANGES (require re-indexing)
- Validation MUST occur at multiple points (initialization, query, indexing)
