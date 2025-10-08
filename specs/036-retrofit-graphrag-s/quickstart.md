# Quickstart: Testing Pipeline Retrofits

**Feature**: 036-retrofit-graphrag-s
**Purpose**: Validate testing infrastructure for BasicRAG, CRAG, BasicRerankRAG, and PyLateColBERT pipelines

---

## Prerequisites

1. **Python Environment**: Python 3.12+
2. **IRIS Database**: Running IRIS instance (Community or Enterprise)
3. **Dependencies**: pytest, pytest-mock installed
4. **API Keys**: OpenAI/Anthropic keys configured (for integration tests)

---

## Quick Test Execution

### 1. Run Contract Tests (Fast, <30 seconds)

```bash
# All contract tests
make test-contract

# Pipeline-specific contract tests
pytest tests/contract/test_basic_rag_contract.py -v
pytest tests/contract/test_crag_contract.py -v
pytest tests/contract/test_basic_rerank_contract.py -v
pytest tests/contract/test_pylate_colbert_contract.py -v
```

**Expected Output**:
```
tests/contract/test_basic_rag_contract.py::test_query_method PASSED
tests/contract/test_basic_rag_contract.py::test_load_documents_method PASSED
tests/contract/test_basic_rag_contract.py::test_query_validates_inputs PASSED
... (30+ tests passing in <30s)
```

---

### 2. Run Error Handling Tests

```bash
# All error handling tests
pytest tests/contract/test_basic_error_handling.py -v
pytest tests/contract/test_crag_error_handling.py -v
pytest tests/contract/test_basic_rerank_error_handling.py -v
pytest tests/contract/test_pylate_colbert_error_handling.py -v
```

**Expected Output**:
```
tests/contract/test_basic_error_handling.py::test_missing_api_key_error PASSED
tests/contract/test_basic_error_handling.py::test_database_connection_retries PASSED
tests/contract/test_basic_error_handling.py::test_error_includes_context PASSED
... (all error handling tests passing)
```

---

### 3. Run Fallback Mechanism Tests

```bash
# Fallback tests (CRAG, BasicRerankRAG, PyLateColBERT)
pytest tests/contract/test_crag_fallback.py -v
pytest tests/contract/test_basic_rerank_fallback.py -v
pytest tests/contract/test_pylate_colbert_fallback.py -v
```

**Expected Output**:
```
tests/contract/test_crag_fallback.py::test_fallback_retrieves_documents PASSED
tests/contract/test_crag_fallback.py::test_fallback_activation_logged PASSED
tests/contract/test_crag_fallback.py::test_fallback_preserves_semantics PASSED
... (all fallback tests passing)
```

---

### 4. Run Dimension Validation Tests

```bash
# Dimension validation tests
pytest tests/contract/test_basic_dimension_validation.py -v
pytest tests/contract/test_crag_dimension_validation.py -v
pytest tests/contract/test_basic_rerank_dimension_validation.py -v
pytest tests/contract/test_pylate_colbert_dimension_validation.py -v
```

**Expected Output**:
```
tests/contract/test_basic_dimension_validation.py::test_query_embedding_is_384d PASSED
tests/contract/test_basic_dimension_validation.py::test_dimension_validation_before_search PASSED
tests/contract/test_basic_dimension_validation.py::test_dimension_mismatch_error PASSED
... (all dimension tests passing)
```

---

### 5. Run Integration Tests (Slower, requires IRIS)

```bash
# All integration tests
pytest tests/integration/ -v -m requires_database

# Pipeline-specific integration tests
pytest tests/integration/test_basic_rag_e2e.py -v
pytest tests/integration/test_crag_e2e.py -v
pytest tests/integration/test_basic_rerank_e2e.py -v
pytest tests/integration/test_pylate_colbert_e2e.py -v
```

**Expected Output**:
```
tests/integration/test_basic_rag_e2e.py::test_full_query_path PASSED
tests/integration/test_basic_rag_e2e.py::test_document_loading_workflow PASSED
tests/integration/test_basic_rag_e2e.py::test_response_quality_metrics PASSED
... (all integration tests passing in ~2 minutes)
```

---

## Validation Checklist

### Contract Test Validation

**BasicRAG**:
- [ ] `test_query_method` passes (validates query API)
- [ ] `test_load_documents_method` passes (validates document loading)
- [ ] `test_query_validates_inputs` passes (input validation)
- [ ] `test_missing_api_key_error` passes (error handling)
- [ ] `test_dimension_validation` passes (384D validation)

**CRAG**:
- [ ] `test_query_method` passes (validates query API)
- [ ] `test_relevance_evaluator_fallback` passes (evaluator fallback)
- [ ] `test_fallback_activation_logged` passes (fallback logging)
- [ ] `test_error_chain_logged` passes (error chain handling)
- [ ] `test_dimension_validation` passes (384D validation)

**BasicRerankRAG**:
- [ ] `test_query_method` passes (validates query API)
- [ ] `test_reranker_fallback` passes (reranker fallback)
- [ ] `test_fallback_preserves_semantics` passes (semantic preservation)
- [ ] `test_reranker_timeout_error` passes (timeout handling)
- [ ] `test_dimension_validation` passes (384D validation)

**PyLateColBERT**:
- [ ] `test_query_method` passes (validates query API)
- [ ] `test_colbert_fallback` passes (ColBERT → dense vector fallback)
- [ ] `test_colbert_dimension_validation` passes (token embedding validation)
- [ ] `test_colbert_model_loading_error` passes (model error handling)
- [ ] `test_fallback_activation_logged` passes (fallback logging)

---

### Integration Test Validation

**All Pipelines**:
- [ ] Document loading → embedding → storage workflow succeeds
- [ ] Query → retrieval → ranking → generation path succeeds
- [ ] Response includes required fields (answer, contexts, metadata)
- [ ] Source attribution is present
- [ ] Context count ≥ 1 (for relevant queries)
- [ ] Execution time logged in metadata

---

## Troubleshooting

### Issue: Contract tests timeout

**Symptom**: Tests exceed 30-second CI/CD limit
**Cause**: Network delays, slow model loading, or database connection issues
**Fix**:
```bash
# Check IRIS database is running
docker ps | grep iris

# Verify IRIS connectivity
python evaluation_framework/test_iris_connectivity.py

# Use lightweight test data
pytest tests/contract/ -v -k "not slow"
```

---

### Issue: Dimension validation tests fail

**Symptom**: `DimensionMismatchError: Expected 384, got 768`
**Cause**: Wrong embedding model configured or database has stale embeddings
**Fix**:
```bash
# Verify embedding model in config
cat config/pipelines.yaml | grep embedding_model

# Should be: "sentence-transformers/all-MiniLM-L6-v2"
# If wrong, update config and re-index:
python scripts/reindex_documents.py --model all-MiniLM-L6-v2
```

---

### Issue: Fallback tests fail

**Symptom**: `AssertionError: Expected fallback_used=True, got False`
**Cause**: Mocking not working or primary method not failing
**Fix**:
```python
# Check mock is correctly patching the method
# Verify method name matches pipeline implementation
mocker.patch.object(
    pipeline,
    '_correct_method_name',  # Must match actual method name
    side_effect=PrimaryMethodError("Simulated failure")
)
```

---

### Issue: Integration tests fail with API errors

**Symptom**: `ConfigurationError: OPENAI_API_KEY not set`
**Cause**: API keys not configured
**Fix**:
```bash
# Set API keys in environment
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Or add to .env file
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Re-run tests
pytest tests/integration/ -v
```

---

## Performance Benchmarks

**Target Performance** (FR-005):
- Contract tests: <30 seconds (CI/CD compatible)
- Integration tests: <2 minutes
- Full test suite: <5 minutes

**Actual Performance** (example):
```
Contract tests:      24.3s  (✅ Under 30s)
Error handling:      12.8s  (✅ Under 30s)
Fallback tests:      18.5s  (✅ Under 30s)
Dimension tests:     9.2s   (✅ Under 30s)
Integration tests:   1m 47s (✅ Under 2m)
Total:               4m 52s (✅ Under 5m)
```

---

## Success Criteria

**Phase 1 Success** (Contract Tests):
- ✅ All contract tests pass (<30s execution)
- ✅ Error handling tests validate clear messages
- ✅ Fallback tests verify graceful degradation
- ✅ Dimension tests validate model compatibility

**Phase 2 Success** (Integration Tests):
- ✅ E2E workflows complete successfully
- ✅ Response quality meets thresholds
- ✅ All pipelines functional end-to-end
- ✅ Integration tests complete <2 minutes

**Phase 3 Success** (Documentation):
- ✅ Quickstart guide validated (this document)
- ✅ Error messages actionable
- ✅ Troubleshooting guide covers common issues
- ✅ Performance benchmarks met

---

## Next Steps

1. **Run full test suite**: `make test-all-pipelines`
2. **Review test coverage**: `pytest --cov=iris_rag tests/`
3. **Validate against requirements**: Check FR-001 through FR-028
4. **Update pipeline docs**: Document new testing patterns
5. **Create pull request**: Include test results and coverage report

---

## Reference

- **Feature Spec**: [spec.md](./spec.md)
- **Research**: [research.md](./research.md)
- **Data Model**: [data-model.md](./data-model.md)
- **Contracts**: [contracts/](./contracts/)
- **Tasks**: [tasks.md](./tasks.md) (generated via `/tasks`)
