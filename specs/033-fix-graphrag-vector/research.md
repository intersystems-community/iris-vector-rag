# Phase 0: Research Findings

**Feature**: Fix GraphRAG Vector Retrieval Logic
**Date**: 2025-10-06
**Status**: Complete

## Research Questions & Answers

### 1. Why does vector search return 0 results?

**Question**: Despite having 2,376 documents with valid 384D embeddings in RAG.SourceDocuments, GraphRAG vector search returns 0 results for all queries. What is the root cause?

**Findings from Feature 032 Investigation**:
- ✅ Schema is healthy: RAG.SourceDocuments exists with 2,376 documents
- ✅ All documents have non-null embeddings (384D from all-MiniLM-L6-v2)
- ✅ Knowledge graph populated: 22,305 entities, 90,298 relationships
- ❌ Logs consistently show "Vector search returned 0 results" for all queries
- ❌ RAGAS shows 0% context precision, 0% context recall

**Evidence**:
```
2025-10-06 21:58:10,498 - INFO - Vector search returned 0 results
2025-10-06 21:58:10,500 - INFO - Hybrid GraphRAG query completed - 0 docs via hybrid_fusion
```

**Database Verification**:
```python
# Feature 032 confirmed:
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL')
# Result: 2376 documents with embeddings

cursor.execute('SELECT LENGTH(content), embedding IS NOT NULL FROM RAG.SourceDocuments LIMIT 3')
# Result: All documents have content and embeddings
```

**Hypothesis**: Vector search SQL query or embedding comparison logic in GraphRAGPipeline is broken, despite working correctly in BasicRAGPipeline with the same database and embeddings.

---

### 2. What IRIS VECTOR_DOT_PRODUCT syntax is correct?

**Question**: How should IRIS VECTOR_DOT_PRODUCT be used for vector similarity search? What patterns are proven to work?

**Research Approach**:
- Review existing working pipelines (BasicRAG, CRAG) for proven patterns
- Examine `common/vector_sql_utils.py` for established helpers
- Understand IRIS-specific SQL syntax requirements

**Expected Findings** (to be validated during investigation tasks):
```python
# Proven pattern from BasicRAG (hypothetical - to be confirmed):
query_vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
sql = f"""
    SELECT TOP ? id, text_content,
           VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?, 'DECIMAL')) as score
    FROM RAG.SourceDocuments
    WHERE embedding IS NOT NULL
    ORDER BY score DESC
"""
cursor.execute(sql, [top_k, query_vector_str])
```

**Key IRIS Requirements** (from documentation):
- VECTOR_DOT_PRODUCT requires specific TO_VECTOR conversion
- Query vector must be formatted as string array
- Parameterized queries must use proper placeholders
- Results must be ordered by score DESC

**Reference Files**:
- `common/vector_sql_utils.py` - proven vector search helpers
- `common/iris_sql_utils.py` - parameterized query helpers
- `iris_rag/pipelines/basic.py` - working vector search implementation

---

### 3. How to validate embedding dimensions match?

**Question**: How should we validate that query embeddings and document embeddings have matching dimensions (384D)?

**Requirement**: FR-005 mandates dimension validation before querying to prevent silent failures.

**Validation Approach**:
```python
# Step 1: Generate query embedding
query_embedding = embedding_model.encode(query)
query_dims = len(query_embedding)

# Step 2: Validate expected dimension
EXPECTED_DIMS = 384  # all-MiniLM-L6-v2 model
if query_dims != EXPECTED_DIMS:
    raise DimensionMismatchError(
        f"Query embedding has {query_dims} dimensions, expected {EXPECTED_DIMS}. "
        f"Verify embedding model is 'sentence-transformers/all-MiniLM-L6-v2'."
    )

# Step 3: Sample check document embeddings (optional validation)
cursor.execute("SELECT embedding FROM RAG.SourceDocuments WHERE embedding IS NOT NULL LIMIT 1")
sample_embedding = cursor.fetchone()[0]
doc_dims = len(sample_embedding)
if doc_dims != EXPECTED_DIMS:
    raise DimensionMismatchError(
        f"Document embeddings have {doc_dims} dimensions, expected {EXPECTED_DIMS}. "
        f"Database may contain embeddings from different model. Re-indexing required."
    )
```

**Error Message Requirements** (FR-005):
- MUST include both query and document dimensions
- MUST suggest actionable fix (model verification, re-indexing)
- MUST raise exception (no silent failures per Constitution VII)

---

### 4. What diagnostic logging is needed?

**Question**: When vector search returns 0 results, what diagnostic information should be logged? (FR-004 requirement)

**Diagnostic Logging Requirements**:
```python
if len(retrieved_docs) == 0:
    logger.info("Vector search returned 0 results")
    logger.debug(f"Query embedding dimensions: {len(query_embedding)}")
    logger.debug(f"Total documents in RAG.SourceDocuments: {total_doc_count}")
    logger.debug(f"Documents with embeddings: {docs_with_embeddings}")
    logger.debug(f"SQL query executed: {sql}")
    logger.debug(f"Top-K parameter: {top_k}")
    logger.debug(f"Sample similarity scores: {sample_scores if sample_scores else 'None returned'}")
```

**Logged Information** (FR-004):
1. **Query embedding dimensions** - verify 384D
2. **Total documents** - ensure data exists
3. **SQL query** - verify VECTOR_DOT_PRODUCT syntax
4. **Top-K parameter** - confirm retrieval limit
5. **Similarity scores** - check if any scores computed (even if below threshold)

**Log Levels**:
- INFO: High-level status ("0 results returned")
- DEBUG: Detailed diagnostics (dims, counts, SQL, scores)

---

## Consolidated Research Findings

### Decision: Fix Vector Search in GraphRAG Pipeline

**Approach**:
1. **Compare implementations**: Review working BasicRAG vector search vs broken GraphRAG vector search
2. **Identify differences**: Find discrepancies in VECTOR_DOT_PRODUCT SQL usage or query construction
3. **Apply proven patterns**: Reuse established patterns from `common/vector_sql_utils.py` and BasicRAG
4. **Add validation**: Implement dimension validation (FR-005)
5. **Add diagnostics**: Implement diagnostic logging (FR-004)
6. **Make configurable**: Expose top-K parameter (FR-006, default K=10)

**Rationale**:
- ✅ BasicRAG works with same database and embeddings → proof that IRIS vector search is functional
- ✅ GraphRAG must have implementation-specific bug → isolated scope for fix
- ✅ Reusing proven patterns minimizes risk → no need to invent new approaches
- ✅ TDD with contracts ensures fix is correct → contract tests written before implementation

**Alternatives Considered**:
- ❌ **Rebuild embeddings**: Unnecessary - embeddings are valid (2,376 documents verified)
- ❌ **Change schema**: Unnecessary - schema is correct (Feature 032 validated)
- ❌ **Modify IRIS configuration**: Unnecessary - BasicRAG proves IRIS works
- ✅ **Fix retrieval logic**: Minimal, targeted fix with proven patterns

---

## Investigation Tasks (Phase 2)

The following investigation tasks will validate these research findings:

**T005**: Compare GraphRAG vector search implementation with BasicRAG vector search
**T006**: Identify specific differences in VECTOR_DOT_PRODUCT SQL usage
**T007**: Review `common/vector_sql_utils.py` for proven patterns and helpers
**T008**: Document root cause findings in investigation notes

**Expected Outcome**: Root cause identification of exact SQL or logic difference causing 0 results in GraphRAG.

---

## References

- **Feature 032 Investigation**: `/Users/tdyar/ws/rag-templates/specs/032-investigate-graphrag-data/investigation/FINDINGS.md`
- **RAGAS Evaluation Results**: `outputs/reports/ragas_evaluations/simple_ragas_report_20251006_215810.json`
- **GraphRAG Pipeline**: `iris_rag/pipelines/graphrag.py`
- **BasicRAG Pipeline**: `iris_rag/pipelines/basic.py`
- **Vector SQL Utils**: `common/vector_sql_utils.py`
- **IRIS SQL Utils**: `common/iris_sql_utils.py`

---

**Research Phase Status**: ✅ COMPLETE
**Next Phase**: Design & Contracts (data-model.md, quickstart.md, contracts/)
