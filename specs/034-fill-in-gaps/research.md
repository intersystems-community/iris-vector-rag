# Research: Testing Strategy for HybridGraphRAG Query Paths

**Feature**: 034-fill-in-gaps
**Date**: 2025-10-07
**Status**: Complete

## Technical Context Resolution

All technical context items were already resolved - no NEEDS CLARIFICATION markers existed in the plan. This section documents the key technical decisions and patterns to be followed.

## Decision 1: Testing Framework and Strategy

**Decision**: Use pytest with contract tests (@pytest.mark.requires_database) and integration tests, leveraging mocking for iris_graph_core fallback scenarios.

**Rationale**:
- Existing test infrastructure uses pytest extensively
- Constitution requires TDD with live IRIS database for all data operations
- Contract tests validate specific functional requirements in isolation
- Integration tests validate end-to-end workflows
- Mocking iris_graph_core allows testing fallback paths without full dependency

**Alternatives Considered**:
- unittest framework: Rejected - less idiomatic for modern Python, existing codebase uses pytest
- Full iris_graph_core integration: Rejected - optional dependency, tests must work without it
- Pure unit tests without database: Rejected - violates Constitution III requirement for live IRIS validation

**Implementation Approach**:
- Contract tests: One test file per query method group (hybrid, rrf, text, vector, kg)
- Mocking strategy: Use pytest-mock to simulate iris_graph_core 0-result and exception scenarios
- Fixtures: Leverage existing conftest.py fixtures (graphrag_pipeline, config_manager)
- Markers: Use @pytest.mark.requires_database for tests needing live IRIS connection

## Decision 2: Test Organization by Functional Requirements

**Decision**: Organize tests by FR groups (Hybrid Fusion FR-001-003, RRF FR-004-006, etc.) with one contract test file per group.

**Rationale**:
- Clear mapping between test files and functional requirements
- Easy to verify coverage of all 28 FRs
- Supports parallel test execution (different files can run concurrently)
- Maintains single responsibility per test file

**Alternatives Considered**:
- Single large test file: Rejected - would be >1000 lines, violates modularity
- One file per FR: Rejected - 28 files is excessive, many FRs are closely related
- Group by implementation file: Rejected - tests organized by behavior, not implementation

**Test File Structure**:
```
tests/contract/
├── test_hybrid_fusion_contract.py         # FR-001 to FR-003 (3 tests)
├── test_rrf_contract.py                   # FR-004 to FR-006 (3 tests)
├── test_text_search_contract.py           # FR-007 to FR-009 (3 tests)
├── test_hnsw_vector_contract.py           # FR-010 to FR-012 (3 tests)
├── test_kg_traversal_contract.py          # FR-013 to FR-015 (3 tests)
├── test_fallback_mechanism_contract.py    # FR-016 to FR-019 (4 tests)
├── test_dimension_validation_contract.py  # FR-020 to FR-022 (existing from F033)
└── test_error_handling_contract.py        # FR-023 to FR-025 (3 tests)

tests/integration/
└── test_hybridgraphrag_e2e.py             # FR-026 to FR-028 (3 tests)
```

## Decision 3: Mocking Strategy for iris_graph_core

**Decision**: Use pytest-mock to patch iris_graph_core methods and simulate fallback scenarios (0 results, exceptions).

**Rationale**:
- iris_graph_core is optional dependency - tests must work without it installed
- Fallback logic is critical path that must be validated
- Mocking allows deterministic testing of error scenarios
- Can test both "happy path" (when available) and "fallback path" (when unavailable/failing)

**Alternatives Considered**:
- Require iris_graph_core for tests: Rejected - makes optional dependency mandatory for CI
- Integration tests only (no mocking): Rejected - can't reliably trigger failure scenarios
- Stub implementation of iris_graph_core: Rejected - overly complex, mocking is simpler

**Mocking Patterns**:
```python
# Pattern 1: Mock 0 results
def test_hybrid_fusion_fallback_on_zero_results(mocker, graphrag_pipeline):
    mock_retrieval = mocker.patch.object(
        graphrag_pipeline.retrieval_methods,
        'retrieve_via_hybrid_fusion',
        return_value=([], 'hybrid_fusion')
    )
    result = graphrag_pipeline.query("test query", method="hybrid")
    assert result.metadata['retrieval_method'] == 'vector_fallback'

# Pattern 2: Mock exception
def test_hybrid_fusion_fallback_on_exception(mocker, graphrag_pipeline):
    mocker.patch.object(
        graphrag_pipeline.retrieval_methods,
        'retrieve_via_hybrid_fusion',
        side_effect=Exception("iris_graph_core connection failed")
    )
    result = graphrag_pipeline.query("test query", method="hybrid")
    assert result.metadata['retrieval_method'] == 'vector_fallback'
```

## Decision 4: Fixture Reuse and Extension

**Decision**: Leverage existing fixtures from Feature 033 (graphrag_pipeline, config_manager, embedding_manager) and add new fixtures for mocking scenarios.

**Rationale**:
- DRY principle - reuse existing validated fixtures
- Consistency with existing test suite
- Simplifies test authoring

**Alternatives Considered**:
- Create all new fixtures: Rejected - duplicates existing infrastructure
- No fixtures, inline setup: Rejected - violates DRY, reduces test readability

**New Fixtures Needed**:
```python
@pytest.fixture
def mock_iris_graph_core_unavailable(mocker):
    """Simulate iris_graph_core not installed."""
    mocker.patch('iris_rag.pipelines.hybrid_graphrag.IRIS_GRAPH_CORE_AVAILABLE', False)

@pytest.fixture
def mock_zero_results_retrieval(mocker):
    """Mock retrieval methods to return 0 results."""
    # Pattern to be reused across tests
```

## Decision 5: Test Data Strategy

**Decision**: Use existing test data from Feature 033 (2,376 documents with 384D embeddings in RAG.SourceDocuments).

**Rationale**:
- Tests validate existing HybridGraphRAG implementation from Feature 033
- No new data ingestion required
- Consistent with RAGAS evaluation data
- Tests query processing paths, not data loading

**Alternatives Considered**:
- Create new test dataset: Rejected - unnecessary, existing data sufficient
- Mock all data operations: Rejected - violates Constitution requirement for live IRIS validation

## Decision 6: Test Execution Performance

**Decision**: Target <5 minute total execution time with parallel test execution where possible.

**Rationale**:
- Constitution emphasizes developer productivity
- CI/CD pipelines require reasonable test execution times
- Different test files can execute in parallel (marked with [P] in tasks)

**Alternatives Considered**:
- No performance target: Rejected - tests could become bottleneck
- <1 minute target: Rejected - unrealistic for 28 tests with live IRIS database

**Performance Strategies**:
- Parallel execution: Use `pytest -n auto` for independent test files
- Shared fixtures: Reuse pipeline instances across tests in same file
- Scoped fixtures: Use session-scoped IRIS connection when possible

## Constitutional Compliance Review

**Framework-First Architecture**: ✓ Tests validate framework components
**Pipeline Validation**: ✓ Tests validate requirement checking and setup orchestration
**Test-Driven Development**: ✓ Writing comprehensive tests IS the feature
**Performance & Scale**: ✓ Tests validate sequential queries and enterprise scale compatibility
**Production Readiness**: ✓ Tests validate logging, error handling, Docker compatibility
**Explicit Error Handling**: ✓ Tests validate no silent failures, clear exceptions
**Standardized Interfaces**: ✓ Tests use existing IRISVectorStore utilities

## Key Patterns to Follow

1. **Test Naming**: `test_{method}_{scenario}` format (e.g., `test_hybrid_fusion_fallback_on_zero_results`)
2. **Assertions**: Use descriptive assertion messages referencing FR numbers
3. **Fixtures**: Parameterize fixtures for different query methods when behavior is identical
4. **Markers**: Use @pytest.mark.requires_database for all tests needing IRIS
5. **Documentation**: Each test includes docstring mapping to specific FR
6. **Mocking**: Mock at method boundary (retrieval_methods.retrieve_via_*), not internal implementation

## References

- Feature 033 implementation: `/Users/tdyar/ws/rag-templates/iris_rag/pipelines/hybrid_graphrag.py`
- Existing contract tests: `/Users/tdyar/ws/rag-templates/tests/contract/test_*_contract.py`
- Pytest fixtures: `/Users/tdyar/ws/rag-templates/tests/conftest.py`
- Constitution: `/Users/tdyar/ws/rag-templates/.specify/memory/constitution.md`
