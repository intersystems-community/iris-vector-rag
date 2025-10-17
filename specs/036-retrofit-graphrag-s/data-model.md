# Data Model: Testing Infrastructure Entities

**Feature**: 036-retrofit-graphrag-s
**Date**: 2025-10-08
**Purpose**: Define test infrastructure entities and their relationships

---

## Overview

This feature does NOT introduce new production entities. It defines test infrastructure entities (fixtures, mocks, test configurations) used to validate the 4 target RAG pipelines.

---

## 1. Test Entities

### 1.1 Pipeline (Test Subject)
**Type**: Fixture entity
**Description**: RAG pipeline instance under test

**Attributes**:
- `pipeline_type`: str - "basic", "crag", "basic_rerank", or "pylate_colbert"
- `config`: ConfigurationManager - Pipeline configuration
- `vector_store`: IRISVectorStore - Vector storage instance
- `llm_func`: Callable - LLM generation function
- `embedding_manager`: EmbeddingManager - Embedding service

**Relationships**:
- Has one ConfigurationManager
- Has one IRISVectorStore
- Has one EmbeddingManager
- Has optional LLM function

**States**:
- `uninitialized`: Before create_pipeline() call
- `initialized`: After successful validation
- `ready`: After load_documents() completes
- `failed`: Initialization validation failed

**Validation Rules**:
- MUST extend RAGPipeline base class
- MUST pass requirement validation when `validate_requirements=True`
- MUST implement query(), load_documents(), embed() methods

---

### 1.2 ContractTest
**Type**: Test case entity
**Description**: Test validating API behavior contracts

**Attributes**:
- `test_name`: str - Descriptive test name
- `functional_requirement`: str - FR-XXX being validated
- `pipeline_type`: str - Target pipeline
- `test_method`: str - Pipeline method being tested (query, load_documents, embed)
- `expected_behavior`: dict - Expected response structure
- `error_conditions`: list[str] - Error scenarios tested

**Relationships**:
- Validates one Pipeline instance
- Maps to one or more Functional Requirements
- Uses zero or more Mock objects

**Validation Rules**:
- Test name MUST follow pattern `test_{method}_{behavior}`
- MUST include FR-XXX in docstring
- MUST use Given-When-Then format in docstring
- MUST complete in <30 seconds

---

### 1.3 IntegrationTest
**Type**: Test case entity
**Description**: End-to-end test validating full query path

**Attributes**:
- `test_name`: str - Descriptive test name
- `pipeline_type`: str - Target pipeline
- `test_data`: list[Document] - Test documents to load
- `query`: str - Test query string
- `expected_context_count`: int - Minimum expected contexts
- `quality_metrics`: dict - Expected quality thresholds

**Relationships**:
- Tests one Pipeline instance
- Loads multiple TestDocument entities
- Validates ResponseQuality

**States**:
- `setup`: Loading test data
- `indexing`: Embedding and storing documents
- `querying`: Executing test query
- `validating`: Asserting quality metrics
- `complete`: Test passed

**Validation Rules**:
- MUST use `@pytest.mark.requires_database`
- MUST validate document loading → embedding → storage → retrieval → generation path
- MUST assert on response quality (context count, source attribution)

---

### 1.4 DiagnosticError (Test Validation Target)
**Type**: Exception entity
**Description**: Enhanced error with context and guidance

**Attributes**:
- `error_type`: str - Error classification (ConfigurationError, DimensionMismatchError, etc.)
- `error_message`: str - Human-readable error description
- `context`: dict - Pipeline state at error time
- `expected_value`: Any - What should have been
- `actual_value`: Any - What was encountered
- `actionable_fix`: str - Steps to resolve

**Relationships**:
- Raised by Pipeline methods
- Logged by DiagnosticLogger
- Validated by ErrorHandlingTest

**Validation Rules**:
- Message MUST include context (pipeline type, operation, state)
- Message MUST include both expected and actual values (when applicable)
- Message MUST suggest actionable fix

**Example Structure**:
```python
DimensionMismatchError(
    message="Embedding dimension mismatch",
    context={
        "pipeline": "BasicRAG",
        "operation": "vector_search",
        "method": "query"
    },
    expected_value=384,
    actual_value=768,
    actionable_fix="Verify embedding model in config.yaml or re-index documents"
)
```

---

### 1.5 FallbackMechanism (Test Validation Target)
**Type**: Behavior entity
**Description**: Automatic recovery strategy when primary method fails

**Attributes**:
- `primary_method`: str - Primary retrieval method name
- `fallback_method`: str - Fallback retrieval method name
- `trigger_condition`: str - What triggers fallback
- `success_indicator`: str - Metadata field indicating fallback used
- `logging_message`: str - Log message when fallback activates

**Relationships**:
- Implemented by Pipeline
- Triggered by specific error conditions
- Logged by DiagnosticLogger
- Validated by FallbackMechanismTest

**Pipeline-Specific Instances**:

**BasicRAG**:
- Primary: live embedding service
- Fallback: cached embeddings
- Trigger: EmbeddingServiceUnavailableError

**CRAG**:
- Primary: relevance-evaluated vector search
- Fallback: vector + text fusion (skip evaluation)
- Trigger: RelevanceEvaluatorError, EvaluatorTimeout

**BasicRerankRAG**:
- Primary: cross-encoder reranking
- Fallback: initial vector similarity ranking
- Trigger: RerankerModelError, RerankerTimeout

**PyLateColBERT**:
- Primary: ColBERT late interaction search
- Fallback: dense vector search (all-MiniLM-L6-v2)
- Trigger: ColBERTModelError, ColBERTScoreError

**Validation Rules**:
- Fallback MUST preserve query semantics (return equivalent results when possible)
- Fallback activation MUST be logged at INFO level
- Fallback method MUST be recorded in response metadata
- System MUST continue functioning after fallback (sustained operation)

---

### 1.6 DimensionValidator (Test Validation Target)
**Type**: Validation component entity
**Description**: Validates embedding dimension compatibility

**Attributes**:
- `expected_dimension`: int - Expected embedding dimension (model-specific)
- `actual_dimension`: int - Actual embedding dimension encountered
- `validation_context`: str - Where validation occurred (query, document)
- `error_raised`: bool - Whether validation failed

**Relationships**:
- Used by Pipeline during query() and embed()
- Raises DimensionMismatchError on failure
- Validated by DimensionValidationTest

**Pipeline-Specific Dimensions**:
- BasicRAG: 384 (all-MiniLM-L6-v2)
- CRAG: 384 (all-MiniLM-L6-v2)
- BasicRerankRAG: 384 query, variable cross-encoder
- PyLateColBERT: Variable (ColBERT token embeddings)

**Validation Rules**:
- MUST validate query embedding dimensions before search
- MUST validate document embedding dimensions during indexing
- Error message MUST include both expected and actual dimensions
- Error message MUST suggest fix (model verification, re-indexing)

---

### 1.7 TestFixture
**Type**: Pytest fixture entity
**Description**: Reusable test setup component

**Attributes**:
- `fixture_name`: str - Fixture function name
- `scope`: str - "session", "module", "function"
- `dependencies`: list[str] - Other fixtures this depends on
- `provides`: Any - Object provided to tests

**Key Fixtures** (to be implemented in conftest.py):

**Pipeline Fixtures** (session-scoped):
```python
basic_rag_pipeline() → BasicRAGPipeline
crag_pipeline() → CRAGPipeline
basic_rerank_pipeline() → BasicRerankRAGPipeline
pylate_colbert_pipeline() → PyLateColBERTPipeline
```

**Utility Fixtures** (function-scoped):
```python
log_capture() → StringIO (for log assertions)
sample_documents() → list[Document] (5 test docs)
sample_query() → str
```

**Shared Fixtures** (already exist):
```python
mocker → pytest-mock mocker
caplog → pytest built-in log capture
```

**Relationships**:
- Fixtures provide Pipeline instances to tests
- Fixtures may depend on other fixtures
- Fixtures manage setup/teardown lifecycle

**Validation Rules**:
- Session-scoped fixtures MUST be idempotent
- Function-scoped fixtures MUST clean up resources
- Fixture names MUST be descriptive and follow convention

---

### 1.8 TestDocument
**Type**: Test data entity
**Description**: Sample document for testing

**Attributes**:
- `doc_id`: str - Unique document identifier
- `content`: str - Document text content
- `metadata`: dict - Document metadata (source, title, etc.)
- `embedding`: list[float] | None - Pre-computed embedding (optional)

**Relationships**:
- Loaded by Pipeline.load_documents()
- Embedded by EmbeddingManager
- Stored in IRISVectorStore
- Retrieved during query tests

**Test Data Sets**:
- **Small** (5 docs): Quick contract tests
- **Medium** (100 docs): Integration tests
- **Large** (1000 docs): RAGAS evaluation only

**Validation Rules**:
- Content MUST be non-empty
- Metadata MUST include source
- Embeddings (if present) MUST match expected dimensions

---

## 2. Entity Relationships Diagram

```
ContractTest ──tests──> Pipeline ──has──> ConfigurationManager
                  │                  │
                  │                  └──> IRISVectorStore
                  │                  │
                  │                  └──> EmbeddingManager
                  │
                  └──uses──> Mock objects
                  │
                  └──validates──> DiagnosticError

IntegrationTest ──tests──> Pipeline
                  │
                  └──loads──> TestDocument
                  │
                  └──validates──> ResponseQuality

FallbackMechanismTest ──validates──> FallbackMechanism
                                      │
                                      └──implemented_by──> Pipeline

DimensionValidationTest ──validates──> DimensionValidator
                                        │
                                        └──used_by──> Pipeline

TestFixture ──provides──> Pipeline
            │
            └──provides──> TestDocument
            │
            └──provides──> log_capture
```

---

## 3. Test Data Schema

### 3.1 Sample Documents (JSON)
```json
{
  "documents": [
    {
      "doc_id": "PMC001",
      "title": "Diabetes Risk Factors",
      "content": "Type 2 diabetes risk factors include obesity...",
      "metadata": {
        "source": "PMC",
        "year": 2023,
        "category": "diabetes"
      }
    }
  ]
}
```

### 3.2 Expected Test Response (Contract)
```python
{
    "answer": str,               # Generated response
    "contexts": list[str],       # Retrieved context chunks
    "metadata": {
        "retrieval_method": str,     # e.g., "vector", "vector_fallback"
        "context_count": int,        # Number of contexts
        "sources": list[str],        # Source document IDs
        "execution_time_ms": float   # Query execution time
    }
}
```

### 3.3 Error Response Structure
```python
{
    "error": str,                # Error type
    "message": str,              # Error description
    "context": {
        "pipeline": str,
        "operation": str,
        "state": dict
    },
    "expected": Any,             # Expected value
    "actual": Any,               # Actual value
    "fix": str                   # Actionable guidance
}
```

---

## 4. Mock Entity Specifications

### 4.1 MockLLMFunction
**Purpose**: Mock LLM API calls for deterministic testing

**Interface**:
```python
def mock_llm(prompt: str) -> str:
    """Return deterministic response based on prompt keywords."""
    if "diabetes" in prompt.lower():
        return "Diabetes is characterized by high blood sugar..."
    return "Generic response for testing"
```

### 4.2 MockEmbeddingService
**Purpose**: Mock embedding service for offline testing

**Interface**:
```python
def mock_embedding(text: str) -> list[float]:
    """Return deterministic 384D embedding."""
    # Use hash of text to generate deterministic vector
    return [0.1] * 384  # Simplified for tests
```

---

## Summary

This data model defines **test infrastructure entities** only. No production database schema changes. The entities support validation of:

1. ✅ API behavior contracts (ContractTest, TestFixture)
2. ✅ Error handling (DiagnosticError)
3. ✅ Fallback mechanisms (FallbackMechanism)
4. ✅ Dimension validation (DimensionValidator)
5. ✅ Integration testing (IntegrationTest, TestDocument)

All entities align with Feature 034 patterns and constitutional testing requirements.
