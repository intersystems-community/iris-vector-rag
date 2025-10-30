# Quickstart: HybridGraphRAG Query Path Testing

**Feature**: 034-fill-in-gaps
**Purpose**: Validate comprehensive test coverage for all HybridGraphRAG query processing paths
**Estimated Time**: 5-10 minutes

## Prerequisites

1. **IRIS Database Running**:
   ```bash
   docker-compose up -d
   # Verify IRIS is healthy
   python evaluation_framework/test_iris_connectivity.py
   ```

2. **Test Data Loaded**:
   ```bash
   # Should have 2,376 documents with embeddings from Feature 033
   # If not, run: make load-data
   ```

3. **Dependencies Installed**:
   ```bash
   source .venv/bin/activate
   pip install pytest pytest-mock
   # Optional: pip install iris-vector-graph
   ```

## Quick Validation

### Step 1: Run Contract Tests (Individual Paths)

```bash
# Test hybrid fusion path (FR-001 to FR-003)
pytest tests/contract/test_hybrid_fusion_contract.py -v

# Test RRF path (FR-004 to FR-006)
pytest tests/contract/test_rrf_contract.py -v

# Test text search path (FR-007 to FR-009)
pytest tests/contract/test_text_search_contract.py -v

# Test HNSW vector path (FR-010 to FR-012)
pytest tests/contract/test_hnsw_vector_contract.py -v

# Test KG traversal path (FR-013 to FR-015)
pytest tests/contract/test_kg_traversal_contract.py -v

# Test fallback mechanism (FR-016 to FR-019)
pytest tests/contract/test_fallback_mechanism_contract.py -v

# Test error handling (FR-023 to FR-025)
pytest tests/contract/test_error_handling_contract.py -v

# Note: test_dimension_validation_contract.py already exists from Feature 033 (FR-020 to FR-022)
```

### Step 2: Run Integration Tests (End-to-End)

```bash
# Test all query methods E2E (FR-026 to FR-028)
pytest tests/integration/test_hybridgraphrag_e2e.py -v
```

### Step 3: Run Full Test Suite

```bash
# Run all new tests for Feature 034
pytest tests/contract/test_*_contract.py tests/integration/test_hybridgraphrag_e2e.py -v

# Run in parallel for faster execution
pytest tests/contract/ tests/integration/ -n auto -v
```

## Expected Results

### Success Criteria

All test cases should **PASS** with the following coverage:

**Contract Tests** (7 files):
- ✅ test_hybrid_fusion_contract.py: 3 tests (FR-001 to FR-003)
- ✅ test_rrf_contract.py: 3 tests (FR-004 to FR-006)
- ✅ test_text_search_contract.py: 3 tests (FR-007 to FR-009)
- ✅ test_hnsw_vector_contract.py: 3 tests (FR-010 to FR-012)
- ✅ test_kg_traversal_contract.py: 3 tests (FR-013 to FR-015)
- ✅ test_fallback_mechanism_contract.py: 4 tests (FR-016 to FR-019)
- ✅ test_dimension_validation_contract.py: 6 tests (FR-020 to FR-022) *existing*
- ✅ test_error_handling_contract.py: 3 tests (FR-023 to FR-025)

**Integration Tests** (1 file):
- ✅ test_hybridgraphrag_e2e.py: 3 tests (FR-026 to FR-028)

**Total**: 25 new tests + 6 existing = **31 tests covering 28 functional requirements**

### Performance Target

- Full test suite execution: **<5 minutes**
- Individual contract test file: **<30 seconds**
- E2E integration tests: **<60 seconds**

## Validation Checklist

After running tests, verify:

- [ ] All 25 new test cases pass
- [ ] Tests execute against live IRIS database (not mocked data)
- [ ] Fallback scenarios correctly trigger vector_fallback
- [ ] All 5 query methods validated (hybrid, rrf, text, vector, kg)
- [ ] Logging output shows diagnostic messages for fallback scenarios
- [ ] Metadata includes retrieval_method, execution_time, num_retrieved
- [ ] No exceptions propagate to user code during error scenarios
- [ ] Pipeline state remains consistent across multiple queries

## Troubleshooting

### Tests Fail: "IRIS database not available"
```bash
# Check IRIS container status
docker ps | grep iris

# Restart IRIS if needed
docker-compose restart iris

# Verify connectivity
python evaluation_framework/test_iris_connectivity.py
```

### Tests Fail: "No documents found in RAG.SourceDocuments"
```bash
# Load test data
make load-data

# Verify data loaded
python -c "
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
config = ConfigurationManager()
conn = ConnectionManager(config).get_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL')
print(f'Documents with embeddings: {cursor.fetchone()[0]}')
"
```

### Tests Fail: "iris_graph_core not found"
```bash
# This is EXPECTED - tests should still pass via fallback
# If you want to test with iris_graph_core:
pip install iris-vector-graph
```

### Tests Timeout
```bash
# Increase pytest timeout
pytest tests/contract/ --timeout=300

# Or run with more verbose output to see where hanging
pytest tests/contract/ -vv -s
```

## Next Steps

After successful test execution:

1. **Review Coverage**: Check that all 28 FRs are validated
2. **Performance Analysis**: Ensure tests execute within time targets
3. **CI Integration**: Tests should run in CI pipeline
4. **Documentation**: Update test documentation if needed

## Related Documentation

- Feature 033: `/Users/intersystems-community/ws/rag-templates/specs/033-fix-graphrag-vector/spec.md`
- HybridGraphRAG Implementation: `iris_rag/pipelines/hybrid_graphrag.py`
- Existing Tests: `tests/contract/`, `tests/integration/`
- Constitution: `.specify/memory/constitution.md`
