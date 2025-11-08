# Research: Testing Framework Fixes

**Feature**: 025-fixes-for-testing
**Date**: 2025-10-03
**Phase**: Phase 0 - Research

## Research Tasks

### 1. pytest Best Practices for IRIS Database Testing

**Decision**: Use module-scoped fixtures for IRIS connections with proper cleanup

**Rationale**:
- IRIS connection setup is expensive (~1-2 seconds per connection)
- Module-scoped fixtures amortize setup cost across all tests in a file
- Proper teardown ensures no connection leaks
- Aligns with constitutional requirement for real database testing

**Alternatives Considered**:
1. **Function-scoped fixtures**: Rejected - too slow (2 seconds × 200 tests = 400 seconds overhead)
2. **Session-scoped fixtures**: Rejected - state pollution risk, harder to isolate test failures
3. **Mock IRIS entirely**: Rejected - violates constitution Section III (requires live database)

**Implementation Pattern**:
```python
@pytest.fixture(scope="module")
def iris_connection(config_manager):
    """Module-scoped IRIS connection fixture."""
    conn_manager = ConnectionManager(config_manager)
    conn = conn_manager.get_connection()
    yield conn
    conn.close()  # Cleanup
```

**pytest-randomly Resolution**:
- **Decision**: Disable globally via `pytest.ini` with `-p no:randomly` flag
- **Rationale**: pytest-randomly calls `numpy.random.seed()` with values that trigger `ValueError: Seed must be between 0 and 2**32 - 1` in numpy/thinc
- **Alternative Considered**: Fix pytest-randomly seed calculation - Rejected (third-party library, not our responsibility)

### 2. Coverage Tools and Reporting

**Decision**: Use pytest-cov with .coveragerc configuration

**Rationale**:
- pytest-cov integrates seamlessly with pytest
- .coveragerc allows excluding test files from coverage calculations
- Supports branch coverage and missing line reports
- Industry standard for Python testing

**Coverage Configuration**:
```ini
[coverage:run]
source = iris_rag,common
omit =
    */tests/*
    */test_*
    */__pycache__/*
    */site-packages/*

[coverage:report]
precision = 1
show_missing = True
skip_covered = False

[coverage:html]
directory = htmlcov
```

**Coverage Targets** (resolved NEEDS CLARIFICATION):
- **Overall**: 60% (current: 10%, industry standard for mature projects)
- **Critical modules**: 80% (pipelines/, storage/, validation/)
- **Non-critical modules**: 40-50% (acceptable for utilities, helpers)

**Alternatives Considered**:
1. **Coverage.py directly**: Rejected - less integration with pytest workflow
2. **Codecov/Coveralls**: Future enhancement for CI/CD, not blocking for local development
3. **Lower targets (40% overall)**: Rejected - insufficient for production RAG framework

### 3. API Contract Testing Patterns

**Decision**: Update tests to match production APIs, not vice versa

**Rationale**:
- Production APIs are stable and used by external consumers
- Tests should validate actual behavior, not desired behavior
- 60 failing tests indicate test assumptions don't match reality
- TDD requires tests to accurately reflect requirements

**Contract Validation Pattern**:
1. Read production code signature (e.g., `BasicRAGPipeline.load_documents`)
2. Update test expectations to match signature
3. Validate test correctly exercises production code path
4. If production API is wrong, fix production FIRST, then update tests

**Example API Mismatch Fix**:
```python
# BEFORE (test assumption):
pipeline.load_documents(documents)  # Expects documents as first positional arg

# ACTUAL (production API):
pipeline.load_documents("", documents=documents)  # documents is kwarg, path is first arg

# FIX (update test to match production):
pipeline.load_documents("", documents=sample_documents)  # ✓ Correct
```

**Alternatives Considered**:
1. **Change production APIs to match tests**: Rejected - breaks existing consumers, violates stability
2. **Create adapter layer**: Rejected - adds unnecessary complexity, hides real API
3. **Skip all failing tests**: Rejected - loses valuable test coverage

### 4. GraphRAG Testing Requirements

**Decision**: Investigate errors, add missing dependencies OR skip with clear explanations

**Rationale**:
- 11 GraphRAG test errors (not failures - import/setup errors)
- Likely missing dependencies or incorrect fixture setup
- Tests should run or skip gracefully, never error
- Constitutional requirement: explicit error handling (no undefined behavior)

**GraphRAG Dependencies to Verify**:
- Entity extraction service (OntologyAwareEntityExtractor)
- Graph storage (IRIS native or via graph-ai integration)
- LLM for entity extraction (OpenAI/Anthropic API keys)
- Graph traversal utilities

**Resolution Options**:
1. **Fix setup** (preferred): Add missing dependencies, configure fixtures properly
2. **Skip with explanation**: If dependencies are optional, skip with `@pytest.mark.skip(reason="GraphRAG requires graph-ai integration")`
3. **Move to integration tests**: If setup is too complex for E2E, move to integration/ with explicit setup docs

**Alternatives Considered**:
1. **Mock entity extraction**: Rejected - defeats purpose of E2E tests per constitution
2. **Remove GraphRAG tests entirely**: Rejected - loses valuable pipeline coverage
3. **Ignore errors**: Rejected - violates explicit error handling principle

### 5. IRIS Vector Store Testing

**Decision**: Validate TO_VECTOR(DOUBLE) behavior with embedded vector strings

**Rationale**:
- IRIS table schema uses `VECTOR(DOUBLE, dimension)` datatype
- TO_VECTOR() function does NOT accept parameter markers (?, :param)
- Vector strings must be embedded directly in SQL: `TO_VECTOR('[0.1,0.2,...]', DOUBLE, 384)`
- Mismatch causes: "Cannot perform vector operation on vectors of different datatypes"

**Test Validation Pattern**:
```python
def test_vector_insertion(iris_vector_store, sample_doc):
    """Validate vector insertion uses DOUBLE datatype."""
    # Insert document with embedding
    iris_vector_store.add_documents([sample_doc])

    # Verify: Query to check vector was stored with DOUBLE type
    cursor = iris_vector_store.connection_manager.get_cursor()
    cursor.execute(f"SELECT VECTOR_TYPE(embedding) FROM {iris_vector_store.table_name} WHERE id = ?", [sample_doc.id])
    vector_type = cursor.fetchone()[0]
    assert vector_type == "DOUBLE", f"Expected DOUBLE, got {vector_type}"
```

**IRIS-Specific Test Requirements**:
1. **Connection validation**: Verify IRIS is running before tests (use `common/iris_port_discovery.py`)
2. **Schema validation**: Verify tables exist with correct schema (VECTOR(DOUBLE, dimension))
3. **Cleanup**: Delete test documents after each test to prevent pollution
4. **Port handling**: Support multiple IRIS ports (11972, 21972, 31972)

**Alternatives Considered**:
1. **Mock vector operations**: Rejected - cannot validate IRIS-specific TO_VECTOR behavior
2. **Use FLOAT datatype**: Rejected - schema uses DOUBLE, mismatch causes errors
3. **Skip vector store tests**: Rejected - core functionality, must be tested

## Summary of Resolved NEEDS CLARIFICATION

From feature spec:

1. **Target Coverage Percentage**: **60% overall** (industry standard for production frameworks)
2. **Critical Module Coverage**: **80% for pipelines, storage, validation** (enterprise requirement)
3. **Test Execution Time Limit**: **< 2 minutes** (current: ~106 seconds, acceptable)

**Additional Resolutions**:
- pytest-randomly: **Disable globally** via `-p no:randomly` in pytest.ini
- API mismatches: **Update tests to match production** (60 failing tests)
- GraphRAG errors: **Investigate and fix OR skip with explanation** (11 errors)
- Vector store: **Validate TO_VECTOR(DOUBLE) behavior** (IRIS-specific)

## Next Steps

Proceed to Phase 1: Design & Contracts
- Create data-model.md (test entities)
- Create contracts/ (test execution, coverage, isolation, API alignment, GraphRAG setup)
- Create quickstart.md (developer testing guide)
- Update CLAUDE.md (agent context for testing patterns)
