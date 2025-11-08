# Phase 1: Quickstart Guide

**Feature**: Fix GraphRAG Vector Retrieval Logic
**Date**: 2025-10-06
**Purpose**: How to validate the fix works correctly

## Prerequisites

### 1. Environment Setup

```bash
# Ensure virtual environment activated
source .venv/bin/activate

# Verify Python version
python --version  # Should be Python 3.12+

# Verify dependencies installed
pip list | grep -E "(iris-rag|sentence-transformers|ragas|langchain)"
```

### 2. Database Running

```bash
# Check IRIS database is running on port 11972
docker ps | grep iris

# Expected output:
# CONTAINER ID   IMAGE                              ...   PORTS                     NAMES
# abc123def456   intersystems/iris-community:2025.3 ...   0.0.0.0:11972->1972/tcp   rag-templates-iris-1

# If not running, start database:
docker-compose up -d
```

### 3. Test Data Loaded

```bash
# Verify documents with embeddings exist
.venv/bin/python -c "
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

config = ConfigurationManager()
conn = ConnectionManager(config).get_connection()
cursor = conn.cursor()

# Check document count
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL')
doc_count = cursor.fetchone()[0]
print(f'Documents with embeddings: {doc_count}')

# Expected output: Documents with embeddings: 2376
if doc_count < 1:
    print('ERROR: No documents with embeddings. Run: make load-data')
else:
    print('✅ Test data ready')
"
```

---

## Quick Smoke Test

### Test 1: Direct Vector Search

```python
# Test vector search directly
from iris_rag import create_pipeline

# Create GraphRAG pipeline with validation
pipeline = create_pipeline("graphrag", validate_requirements=True)

# Query
result = pipeline.query("What are the symptoms of diabetes?")

# Validate
print(f"Contexts retrieved: {len(result.contexts)}")  # Should be >0 after fix
print(f"Answer length: {len(result.answer)}")         # Should be >0
print(f"Answer preview: {result.answer[:100]}...")    # Should not be "No relevant documents"

# Expected after fix:
# Contexts retrieved: 10
# Answer length: 250
# Answer preview: Diabetes mellitus symptoms include increased thirst, frequent urination...
```

**Success Criteria**:
- ✅ `len(result.contexts) > 0` (retrieves documents)
- ✅ Answer is not "No relevant documents found"
- ✅ Answer mentions diabetes symptoms

---

### Test 2: Embedding Dimension Validation

```python
# Verify dimension validation (FR-005)
from iris_rag import create_pipeline
from iris_rag.embeddings.manager import EmbeddingManager

pipeline = create_pipeline("graphrag")
emb_manager = EmbeddingManager(pipeline.config)

# Generate query embedding
query_embedding = emb_manager.generate_embedding("test query")

# Validate dimensions
print(f"Query embedding dimensions: {len(query_embedding)}")  # Should be 384
assert len(query_embedding) == 384, "Query embedding dimension mismatch"

# Success
print("✅ Embedding dimensions validated: 384D")
```

**Success Criteria**:
- ✅ Query embedding is exactly 384 dimensions
- ✅ No DimensionMismatchError raised

---

### Test 3: Top-K Configuration

```python
# Verify top-K is configurable (FR-006)
from iris_rag import create_pipeline

# Default K=10
pipeline_default = create_pipeline("graphrag")
result = pipeline_default.query("What are the symptoms of diabetes?")
print(f"Default K=10: Retrieved {len(result.contexts)} contexts")  # Should be ≤10

# Custom K=5 (via config override)
from iris_rag.config.manager import ConfigurationManager
config = ConfigurationManager()
config.update_config({"retrieval": {"top_k": 5}})
pipeline_custom = create_pipeline("graphrag", config_manager=config)
result = pipeline_custom.query("What are the symptoms of diabetes?")
print(f"Custom K=5: Retrieved {len(result.contexts)} contexts")   # Should be ≤5

# Success
print("✅ Top-K configuration validated")
```

**Success Criteria**:
- ✅ Default retrieval returns ≤10 documents
- ✅ Custom K=5 returns ≤5 documents
- ✅ Configuration is respected

---

## Contract Test Validation

### Run Contract Tests (TDD)

```bash
# Before fix: Contract tests should FAIL
.venv/bin/pytest tests/contract/test_vector_search_contract.py -v

# Expected output (before fix):
# tests/contract/test_vector_search_contract.py::test_vector_search_returns_documents FAILED
# tests/contract/test_vector_search_contract.py::test_results_sorted_by_score FAILED
# tests/contract/test_vector_search_contract.py::test_dimension_validation PASSED
# 2 failed, 1 passed

# After fix: Contract tests should PASS
.venv/bin/pytest tests/contract/test_vector_search_contract.py -v

# Expected output (after fix):
# tests/contract/test_vector_search_contract.py::test_vector_search_returns_documents PASSED
# tests/contract/test_vector_search_contract.py::test_results_sorted_by_score PASSED
# tests/contract/test_vector_search_contract.py::test_dimension_validation PASSED
# tests/contract/test_vector_search_contract.py::test_diagnostic_logging PASSED
# 4 passed
```

**Success Criteria**:
- ✅ All contract tests pass after fix
- ✅ `test_vector_search_returns_documents` passes (FR-001)
- ✅ `test_dimension_validation` passes (FR-005)
- ✅ `test_diagnostic_logging` passes (FR-004)

---

## RAGAS Evaluation

### Run RAGAS Evaluation (Acceptance Test)

```bash
# Set environment variables
export IRIS_HOST=localhost
export IRIS_PORT=11972
export RAGAS_PIPELINES="graphrag"

# Run RAGAS evaluation on GraphRAG pipeline
.venv/bin/python scripts/simple_working_ragas.py

# Wait for completion (may take 2-5 minutes for 5 queries)
# Output: outputs/reports/ragas_evaluations/simple_ragas_report_YYYYMMDD_HHMMSS.json
```

### Validate RAGAS Results

```bash
# Extract GraphRAG metrics from latest report
cat outputs/reports/ragas_evaluations/simple_ragas_report_*.json | \
  python -m json.tool | \
  grep -A 10 '"graphrag"'

# Expected output (after fix):
# "graphrag": {
#   "overall_performance": 0.45,        # >14.4% baseline (IMPROVED)
#   "answer_correctness": 0.35,
#   "faithfulness": 0.65,
#   "context_precision": 0.42,          # >30% target (FR-019 PASS)
#   "context_recall": 0.28,             # >20% target (FR-020 PASS)
#   "answer_relevancy": 0.50,
#   "successful_queries": 5,
#   "failed_queries": 0,
#   "success_rate": 1.0
# }
```

**Success Criteria (FR-019, FR-020, FR-021, FR-022)**:
- ✅ `context_precision > 0.30` (>30% target)
- ✅ `context_recall > 0.20` (>20% target)
- ✅ `overall_performance > 0.144` (improved from 14.4% baseline)
- ✅ `success_rate == 1.0` (all queries retrieve documents)

---

## Diagnostic Logging Verification

### Check Logs for Diagnostic Information (FR-004)

```bash
# Run query with DEBUG logging enabled
LOGLEVEL=DEBUG .venv/bin/python -c "
from iris_rag import create_pipeline
import logging
logging.basicConfig(level=logging.DEBUG)

pipeline = create_pipeline('graphrag')
result = pipeline.query('What are the symptoms of diabetes?')
"

# Expected log output (after fix):
# DEBUG - Query embedding dimensions: 384
# DEBUG - Total documents in RAG.SourceDocuments: 2376
# DEBUG - Documents with embeddings: 2376
# DEBUG - SQL query executed: SELECT TOP 10 id, text_content, VECTOR_DOT_PRODUCT...
# DEBUG - Top-K parameter: 10
# INFO - Vector search returned 10 results
```

**Validation**:
```bash
# If 0 results returned, logs MUST contain (FR-004):
# INFO - Vector search returned 0 results
# DEBUG - Query embedding dimensions: 384
# DEBUG - Total documents in RAG.SourceDocuments: 2376
# DEBUG - SQL query: <actual SQL>
# DEBUG - Top-K parameter: 10
# DEBUG - Sample similarity scores: None returned
```

**Success Criteria**:
- ✅ Logs contain query embedding dimensions
- ✅ Logs contain total document count
- ✅ Logs contain SQL query executed
- ✅ Logs contain top-K parameter value
- ✅ When 0 results, logs explain why (missing data, dimension mismatch, etc.)

---

## Integration Test Suite

### Run Full GraphRAG Integration Tests

```bash
# Run all integration tests for GraphRAG pipeline
.venv/bin/pytest tests/integration/test_graphrag_vector_search.py -v

# Expected tests:
# test_graphrag_vector_search_retrieval PASSED        # FR-001: Returns documents
# test_graphrag_top_k_configuration PASSED            # FR-006: Top-K configurable
# test_graphrag_dimension_validation PASSED           # FR-005: Validates dimensions
# test_graphrag_diagnostic_logging PASSED             # FR-004: Logs diagnostics
# test_graphrag_embedding_consistency PASSED          # FR-003: Works with 384D
```

**Success Criteria**:
- ✅ All integration tests pass
- ✅ Tests cover all functional requirements (FR-001 to FR-006)

---

## End-to-End Validation

### Run E2E GraphRAG Pipeline Test

```bash
# Run comprehensive E2E test
.venv/bin/pytest tests/e2e/test_graphrag_pipeline_e2e.py -v

# Expected workflow:
# 1. Initialize GraphRAG pipeline
# 2. Validate requirements
# 3. Execute queries
# 4. Verify retrieval
# 5. Validate answer quality

# Expected output:
# test_graphrag_pipeline_initialization PASSED
# test_graphrag_query_execution PASSED
# test_graphrag_retrieval_quality PASSED
# test_graphrag_answer_generation PASSED
```

**Success Criteria**:
- ✅ Pipeline initializes without errors
- ✅ All queries return documents
- ✅ Answers are relevant and grounded in retrieved context

---

## Regression Testing

### Verify No Impact on Other Pipelines

```bash
# Ensure BasicRAG still works (no regression)
.venv/bin/python -c "
from iris_rag import create_pipeline
pipeline = create_pipeline('basic')
result = pipeline.query('What are the symptoms of diabetes?')
print(f'BasicRAG contexts: {len(result.contexts)}')  # Should be >0
"

# Ensure CRAG still works
.venv/bin/python -c "
from iris_rag import create_pipeline
pipeline = create_pipeline('crag')
result = pipeline.query('What are the symptoms of diabetes?')
print(f'CRAG contexts: {len(result.contexts)}')  # Should be >0
"

# Ensure HybridGraphRAG still works (uses same fix)
.venv/bin/python -c "
from iris_rag import create_pipeline
pipeline = create_pipeline('hybrid_graphrag')
result = pipeline.query('What are the symptoms of diabetes?')
print(f'HybridGraphRAG contexts: {len(result.contexts)}')  # Should be >0
"
```

**Success Criteria**:
- ✅ BasicRAG retrieval still works (no regression)
- ✅ CRAG retrieval still works (no regression)
- ✅ HybridGraphRAG benefits from same fix (improvement)

---

## Troubleshooting

### Issue: No Documents Retrieved After Fix

```bash
# 1. Check database connectivity
docker ps | grep iris  # Ensure IRIS running

# 2. Verify test data
.venv/bin/python -c "
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
config = ConfigurationManager()
conn = ConnectionManager(config).get_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL')
print(f'Documents: {cursor.fetchone()[0]}')  # Should be 2376
"

# 3. Check diagnostic logs
LOGLEVEL=DEBUG .venv/bin/python scripts/test_graphrag_validation.py 2>&1 | grep -A 5 "Vector search"

# 4. Verify embedding model
.venv/bin/python -c "
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.config.manager import ConfigurationManager
config = ConfigurationManager()
emb_mgr = EmbeddingManager(config)
test_emb = emb_mgr.generate_embedding('test')
print(f'Model dimensions: {len(test_emb)}')  # Should be 384
"
```

### Issue: DimensionMismatchError Raised

```bash
# Error indicates embedding dimension != 384
# Solution: Verify embedding model configuration

# Check config file
cat iris_rag/config/default_config.yaml | grep -A 3 embedding_model

# Expected:
# embedding_model: sentence-transformers/all-MiniLM-L6-v2
# embedding_dimension: 384

# If different model configured, update or re-index documents
```

### Issue: RAGAS Performance Below Target

```bash
# If context_precision <30% or context_recall <20%:

# 1. Check retrieval is working
LOGLEVEL=INFO .venv/bin/python -c "
from iris_rag import create_pipeline
pipeline = create_pipeline('graphrag')
result = pipeline.query('What are the symptoms of diabetes?')
print(f'Retrieved: {len(result.contexts)} contexts')  # Should be >0
"

# 2. Inspect retrieved contexts
# ... (print contexts to verify relevance)

# 3. Review test queries
cat scripts/simple_working_ragas.py | grep -A 20 "test_queries"

# 4. Consider data quality
# May need more/better source documents for specific topics
```

---

**Quickstart Guide Status**: ✅ COMPLETE
**Next**: Contract test specifications in contracts/ directory
