# Research: GraphRAG Testing Pattern Analysis

**Feature**: 036-retrofit-graphrag-s
**Date**: 2025-10-08
**Purpose**: Document GraphRAG testing patterns and their application to BasicRAG, CRAG, BasicRerankRAG, and PyLateColBERT

---

## Overview

Feature 034 established comprehensive testing patterns for HybridGraphRAG pipeline. This research analyzes those patterns and documents how they apply to the 4 target pipelines.

---

## 1. Testing Pattern Categories (from Feature 034)

### 1.1 Contract Tests
**Purpose**: Validate API behavior contracts for core pipeline methods

**Pattern Structure** (from `test_fallback_mechanism_contract.py`):
- Class-based test organization (`Test{Pattern}Contract`)
- Pytest fixtures for pipeline initialization
- Method-level tests with FR requirement traceability
- Given-When-Then documentation in docstrings
- Mock usage for simulating failure conditions
- Assertion of response structure and metadata

**Key Insights**:
- Each test validates ONE functional requirement (FR-XXX)
- Tests use pytest markers (`@pytest.mark.requires_database`)
- Mocking via `pytest-mock` mocker fixture
- Metadata inspection to verify retrieval methods

**Decision**: Apply same structure to all 4 pipelines
**Rationale**: Consistency enables pattern recognition, easier maintenance, and cross-pipeline comparison

---

### 1.2 Error Handling Tests
**Purpose**: Validate graceful degradation and clear error messages

**Pattern Structure** (from `test_error_handling_contract.py`):
- Test missing required resources (tables, connections)
- Test exception propagation (should NOT propagate to caller)
- Test logging of errors (via `caplog` fixture)
- Test fallback activation on errors
- Test sustained functionality after errors

**Key Insights**:
- Error tests verify NO EXCEPTIONS propagate to caller
- Logging validation uses `caplog.text` inspection
- Fallback verification via metadata inspection
- Multiple queries test sustained functionality

**Decision**: Implement error handling tests for each pipeline's failure modes
**Rationale**: Each pipeline has unique dependencies (BasicRAG: embeddings, CRAG: relevance evaluator, PyLateColBERT: ColBERT model)

---

### 1.3 Diagnostic Logging Tests
**Purpose**: Validate contextual logging at INFO and DEBUG levels

**Pattern Structure** (from `test_diagnostic_logging_contract.py`):
- String-based log capture via `StringIO` handler
- Regex pattern matching for log validation
- Differentiated INFO vs DEBUG expectations
- Specific diagnostic items validated (dimensions, query text, result counts)

**Key Insights**:
- Tests validate BOTH INFO and DEBUG log levels
- INFO: High-level status (e.g., "0 results returned")
- DEBUG: Detailed diagnostics (dimensions, SQL queries, top-K)
- Regex used for flexible log format matching

**Decision**: Standardize diagnostic logging across all 4 pipelines
**Rationale**: Consistent logging improves debugging and operational visibility

---

### 1.4 Dimension Validation Tests
**Purpose**: Validate embedding dimension compatibility

**Pattern Structure** (from `test_dimension_validation_contract.py`):
- Test query embedding dimensions (expected: 384 for all-MiniLM-L6-v2)
- Test document embedding dimensions (database validation)
- Test dimension mismatch error messages
- Test actionable error guidance

**Key Insights**:
- Embedding dimension is model-specific (384 for all-MiniLM-L6-v2)
- Database dimension validation via SQL query sampling
- Error messages MUST include both expected and actual dimensions
- Error messages MUST suggest fixes (re-indexing, model verification)

**Decision**: Validate dimensions for each pipeline's embedding model
**Rationale**: Different pipelines may use different embedding models (PyLateColBERT uses ColBERT, others use sentence-transformers)

---

### 1.5 Fallback Mechanism Tests
**Purpose**: Validate automatic recovery when primary methods fail

**Pattern Structure** (from `test_fallback_mechanism_contract.py`):
- Test fallback retrieves documents successfully
- Test fallback logging (method used, reason for fallback)
- Test fallback preserves query semantics
- Test configuration to disable fallbacks

**Key Insights**:
- HybridGraphRAG falls back to IRISVectorStore on graph failures
- Fallback metadata: `retrieval_method` indicates fallback usage
- Logging captures fallback activation reason

**Decision**: Implement fallback tests for pipelines with multiple retrieval methods
**Rationale**: BasicRAG has single retrieval (no fallback needed), CRAG has evaluator → search fallback, HybridGraphRAG has graph → vector fallback

**Alternatives Considered**:
- BasicRAG: No fallback (single retrieval method) - ACCEPTED
- CRAG: Implement evaluator → text search fallback - ACCEPTED
- PyLateColBERT: Implement ColBERT → dense vector fallback - TO BE DESIGNED

---

### 1.6 Integration (E2E) Tests
**Purpose**: Validate full query path end-to-end

**Pattern Structure** (from `test_graphrag_vector_search.py`):
- Test document loading → embedding → storage workflow
- Test query → retrieval → ranking → generation path
- Test response quality metrics (relevance, completeness, sources)
- Test with real IRIS database (not mocked)

**Key Insights**:
- Integration tests use `@pytest.mark.requires_database`
- Tests validate complete pipeline lifecycle
- Response quality assertions (context count, source attribution)

**Decision**: Implement E2E tests for each pipeline
**Rationale**: Validates full pipeline integration, critical for production readiness

---

## 2. Target Pipeline Analysis

### 2.1 BasicRAG Pipeline
**File**: `iris_rag/pipelines/basic.py`
**Core Methods**: `query()`, `load_documents()`, `embed()`
**Dependencies**:
- EmbeddingManager (sentence-transformers)
- IRISVectorStore (vector similarity search)
- LLM function (generation)

**Unique Testing Needs**:
- Embedding service unavailability → cached embedding fallback (FR-016)
- Missing API keys → clear diagnostic errors (FR-009, FR-010)
- Vector dimension validation (FR-021 to FR-024)

**Fallback Strategy**: Embedding service failure → use cached embeddings if available
**Error Scenarios**: Missing LLM API key, missing embedding model, database connection failure

---

### 2.2 CRAG Pipeline
**File**: `iris_rag/pipelines/crag.py`
**Core Methods**: `query()`, `load_documents()`, `evaluate_relevance()`
**Dependencies**:
- EmbeddingManager
- IRISVectorStore
- Relevance evaluator (LLM-based)
- LLM function (generation)

**Unique Testing Needs**:
- Relevance evaluation failure → skip evaluation, proceed with retrieval (FR-012)
- Low relevance score → web search augmentation (CRAG-specific)
- Evaluator timeout → fallback to vector search only

**Fallback Strategy**: Relevance evaluator failure → vector search + text search fusion
**Error Scenarios**: Evaluator API failure, timeout, malformed evaluation response

---

### 2.3 BasicRerankRAG Pipeline
**File**: `iris_rag/pipelines/basic_rerank.py`
**Core Methods**: `query()`, `load_documents()`, `rerank()`
**Dependencies**:
- EmbeddingManager
- IRISVectorStore
- Cross-encoder reranker model
- LLM function (generation)

**Unique Testing Needs**:
- Reranker model loading failure → fallback to initial retrieval ranking (FR-012)
- Reranker dimension mismatch → clear error (FR-021 to FR-024)
- Reranker timeout → return unranked results with warning

**Fallback Strategy**: Reranker failure → use initial vector similarity ranking
**Error Scenarios**: Reranker model missing, reranker timeout, dimension mismatch

---

### 2.4 PyLateColBERT Pipeline
**File**: `iris_rag/pipelines/pylate_colbert.py`
**Core Methods**: `query()`, `load_documents()`, `colbert_search()`
**Dependencies**:
- PyLate ColBERT model
- IRISVectorStore (fallback)
- LLM function (generation)

**Unique Testing Needs**:
- ColBERT model loading failure → fallback to dense vector search (FR-012, FR-015)
- ColBERT dimension validation (late interaction embeddings)
- ColBERT score computation errors

**Fallback Strategy**: ColBERT failure → fall back to dense vector (all-MiniLM-L6-v2)
**Error Scenarios**: ColBERT model missing, dimension mismatch, score computation failure

---

## 3. Test File Naming Convention

**Decision**: Follow Feature 034 naming pattern
**Pattern**: `test_{pipeline_name}_{pattern_type}.py`

**Examples**:
- Contract: `test_basic_rag_contract.py`
- Error handling: `test_basic_error_handling.py`
- Dimension validation: `test_basic_dimension_validation.py`
- Integration: `test_basic_rag_e2e.py`

**Rationale**: Consistent naming enables easy discovery and grep-ability

---

## 4. Pytest Fixture Strategy

**Decision**: Extend `tests/conftest.py` with pipeline-specific fixtures

**Required Fixtures**:
```python
@pytest.fixture
def basic_rag_pipeline():
    return create_pipeline("basic", validate_requirements=True)

@pytest.fixture
def crag_pipeline():
    return create_pipeline("crag", validate_requirements=True)

@pytest.fixture
def basic_rerank_pipeline():
    return create_pipeline("basic_rerank", validate_requirements=True)

@pytest.fixture
def pylate_colbert_pipeline():
    return create_pipeline("pylate_colbert", validate_requirements=True)
```

**Shared Fixtures** (already exist in conftest.py):
- `log_capture`: StringIO-based log capture
- `mocker`: pytest-mock mocker
- `caplog`: pytest built-in log capture

**Rationale**: Session-scoped fixtures improve test performance, consistent initialization

---

## 5. Mock Strategy for External Services

**Decision**: Use `pytest-mock` for service mocking, NO mock for IRIS database

**Mock Targets**:
- LLM API calls (OpenAI, Anthropic) → mock via `mocker.patch`
- Embedding model loading → mock to avoid slow model downloads
- External APIs (CRAG web search) → mock for deterministic tests

**NO MOCK Targets**:
- IRIS database connections → use live database (constitutional requirement)
- Vector operations → use real IRIS vector search
- SQL queries → execute against live database

**Rationale**: Constitution III requires live IRIS database testing. Mocking LLM APIs reduces test brittleness and cost.

---

## 6. CI/CD Performance Requirements

**Decision**: Contract tests MUST complete <30 seconds

**Strategy**:
- Use lightweight test data (5-10 documents)
- Mock expensive LLM calls
- Parallelize independent tests
- Skip slow integration tests in quick CI runs

**Make Targets** (to be added):
```bash
make test-contract          # Run contract tests only (<30s)
make test-integration       # Run integration tests (slower)
make test-all-pipelines     # Run all pipeline tests
```

**Rationale**: FR-005 requires <30s for CI/CD compatibility

---

## 7. Error Message Standards

**Decision**: All error messages MUST follow actionable template

**Template**:
```
Error: {specific_problem}
Context: {pipeline_type}, {operation}, {current_state}
Expected: {what_should_be}
Actual: {what_is}
Fix: {actionable_steps}
```

**Example** (from dimension validation):
```
Error: Embedding dimension mismatch
Context: BasicRAG pipeline, vector search, query embedding
Expected: 384 dimensions (all-MiniLM-L6-v2)
Actual: 768 dimensions
Fix: Verify embedding model configuration in config.yaml
      Or re-index documents with correct model
```

**Rationale**: FR-010 requires actionable guidance in all error messages

---

## 8. Logging Standards

**Decision**: Standardize logging levels across all pipelines

**INFO Level** (user-facing):
- Query execution started/completed
- Document retrieval status (count, method used)
- Fallback activation (with reason)
- Zero results warnings

**DEBUG Level** (developer diagnostics):
- Query embedding dimensions
- SQL queries executed
- Top-K parameters
- Similarity scores
- Total documents in database
- Documents with embeddings count

**ERROR Level** (failures):
- Missing required configuration
- Connection failures
- Dimension mismatches
- Model loading failures

**Rationale**: FR-009, FR-013 require contextual diagnostic logging

---

## 9. Test Data Strategy

**Decision**: Reuse Feature 034 test data (PMC diabetes documents)

**Test Data Sources**:
- Small set (5 docs): `tests/data/sample_pmc_docs.json` → quick contract tests
- Medium set (100 docs): `tests/data/pmc_diabetes_100.json` → integration tests
- Large set (1000 docs): RAGAS evaluation only

**Loading Strategy**:
- Contract tests: Pre-loaded fixture data (fast)
- Integration tests: Load via pipeline.load_documents()
- E2E tests: Full indexing workflow

**Rationale**: Consistent test data enables cross-pipeline comparison

---

## 10. Acceptance Criteria Validation

**Decision**: Map each functional requirement to specific test(s)

**Traceability Matrix** (to be created in contracts/):
```
FR-001: Contract tests validate query/load_documents/embed
  → test_basic_rag_contract.py::test_query_method
  → test_basic_rag_contract.py::test_load_documents_method
  → test_basic_rag_contract.py::test_embed_method

FR-002: Input validation tests
  → test_basic_rag_contract.py::test_query_validates_inputs
  → test_basic_rag_contract.py::test_optional_vs_required_params

[... etc for all 28 FRs]
```

**Rationale**: FR-006 requires clear failure messages with expected vs actual behavior

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Test structure | Class-based pytest with fixtures | Consistency with Feature 034 |
| Naming convention | `test_{pipeline}_{pattern}.py` | Easy discovery and grep |
| Database testing | Live IRIS (no mocking) | Constitutional requirement III |
| LLM/API testing | Mock via pytest-mock | Reduce cost and brittleness |
| Error messages | Actionable template | FR-010 requirement |
| Logging levels | INFO/DEBUG/ERROR standard | FR-013 requirement |
| Test data | Reuse Feature 034 PMC docs | Cross-pipeline consistency |
| CI/CD performance | Contract tests <30s | FR-005 requirement |
| Fixtures | Pipeline-specific + shared | Performance and consistency |
| Fallback testing | Pipeline-specific strategies | Different pipelines, different fallbacks |

---

## Unknowns Resolved

All Technical Context items were specified (no NEEDS CLARIFICATION markers). Research successfully documented:

1. ✅ GraphRAG testing patterns analyzed
2. ✅ Target pipeline characteristics identified
3. ✅ Fallback strategies designed for each pipeline
4. ✅ Error scenarios documented
5. ✅ Logging standards defined
6. ✅ Test data strategy established
7. ✅ CI/CD performance approach defined

**Status**: Research complete. Ready for Phase 1 (Design & Contracts).
