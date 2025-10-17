# Tasks: Retrofit GraphRAG Testing Improvements to Other Pipelines

**Feature**: 036-retrofit-graphrag-s
**Branch**: `036-retrofit-graphrag-s`
**Input**: Design documents from `/Users/tdyar/ws/rag-templates/specs/036-retrofit-graphrag-s/`

## Execution Flow (main)
```
1. Load plan.md: ✅ Python 3.12, pytest, pytest-mock, IRIS database
2. Load design documents:
   ✅ data-model.md: 8 test entities (Pipeline, ContractTest, etc.)
   ✅ contracts/: 4 contract files (API, error, fallback, dimension)
   ✅ research.md: 6 testing patterns, 4 target pipelines
   ✅ quickstart.md: Validation scenarios
3. Generate tasks by category:
   ✅ Setup: Fixtures, test data, pytest config
   ✅ Tests: 16 contract test files + 4 integration test files
   ✅ Validation: Test execution, coverage, FR traceability
4. Task ordering: Setup → Contract Tests → Integration Tests → Validation
5. Tasks numbered: T001-T028
6. Parallel execution: [P] marks independent test files
7. SUCCESS: 28 tasks ready for TDD execution
```

---

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All paths relative to repository root: `/Users/tdyar/ws/rag-templates/`

---

## Phase 3.1: Setup & Test Infrastructure

### T001: ✅ COMPLETE - Update conftest.py with pipeline fixtures
**File**: `tests/conftest.py`
**Dependencies**: None
**Requirements**: FR-001, FR-007

**Implementation**:
1. Add session-scoped fixture `basic_rag_pipeline`:
   ```python
   @pytest.fixture(scope="session")
   def basic_rag_pipeline():
       return create_pipeline("basic", validate_requirements=True)
   ```
2. Add session-scoped fixture `crag_pipeline`
3. Add session-scoped fixture `basic_rerank_pipeline`
4. Add session-scoped fixture `pylate_colbert_pipeline`
5. Add function-scoped fixture `sample_documents`:
   ```python
   @pytest.fixture
   def sample_documents():
       return [
           Document(content="Diabetes symptoms...", metadata={"source": "PMC001"}),
           # 4 more sample documents
       ]
   ```
6. Add function-scoped fixture `sample_query`:
   ```python
   @pytest.fixture
   def sample_query():
       return "What are the symptoms of diabetes?"
   ```

**Acceptance**:
- All 4 pipeline fixtures return valid pipeline instances
- Fixtures use `create_pipeline()` factory with validation
- Sample data fixtures provide 5 test documents
- All fixtures documented with docstrings

---

### T002: ✅ COMPLETE - [P] Create sample test data file
**File**: `tests/data/sample_pmc_docs_basic.json`
**Dependencies**: None

**Implementation**:
1. Create JSON file with 5 sample PMC diabetes documents
2. Structure:
   ```json
   {
     "documents": [
       {
         "doc_id": "PMC001",
         "title": "Diabetes Risk Factors",
         "content": "Type 2 diabetes risk factors include...",
         "metadata": {"source": "PMC", "category": "diabetes"}
       }
     ]
   }
   ```
3. Keep documents short (<200 words each) for fast tests
4. Cover variety: symptoms, treatment, prevention, complications, diagnosis

**Acceptance**:
- 5 documents total
- All documents have doc_id, title, content, metadata
- Total file size <10KB
- Documents relevant to diabetes domain

---

### T003: ✅ COMPLETE - [P] Configure pytest markers for test categorization
**File**: `pytest.ini`
**Dependencies**: None

**Implementation**:
1. Add marker for contract tests:
   ```ini
   markers =
       contract: Contract test validating API behavior
       error_handling: Error handling and diagnostic message tests
       fallback: Fallback mechanism tests
       dimension: Dimension validation tests
       integration: Integration test requiring full pipeline
   ```
2. Verify existing `requires_database` marker present
3. Add marker for pipeline-specific tests:
   ```ini
       basic_rag: Tests for BasicRAG pipeline
       crag: Tests for CRAG pipeline
       basic_rerank: Tests for BasicRerankRAG pipeline
       pylate_colbert: Tests for PyLateColBERT pipeline
   ```

**Acceptance**:
- All markers documented in pytest.ini
- Markers enable selective test execution
- Example: `pytest -m "contract and basic_rag"`

---

## Phase 3.2: Contract Tests - BasicRAG Pipeline (TDD)

### T004: ✅ COMPLETE - [P] BasicRAG API contract test
**File**: `tests/contract/test_basic_rag_contract.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-001, FR-002, FR-003, FR-004
**Marker**: `@pytest.mark.contract`, `@pytest.mark.basic_rag`

**Implementation**:
1. Create `TestBasicRAGContract` class
2. Implement `test_query_method_exists`:
   - Verify pipeline has query() method
   - Verify signature accepts query, method, top_k, **kwargs
3. Implement `test_query_validates_required_parameter`:
   - Test query=None raises ValueError
   - Test query="" raises ValueError
4. Implement `test_query_validates_top_k_range`:
   - Test top_k=0 raises ValueError
   - Test top_k=101 raises ValueError
5. Implement `test_query_returns_valid_structure`:
   - Execute query, verify response has answer, contexts, metadata
   - Verify metadata has retrieval_method, context_count, sources
6. Implement `test_load_documents_method_exists`
7. Implement `test_load_documents_validates_input`:
   - Test documents=[] raises ValueError
   - Test document without content raises ValueError

**Acceptance**:
- All tests follow Given-When-Then format in docstrings
- All tests include FR-XXX traceability
- Tests use `basic_rag_pipeline` fixture
- Test suite executes in <10s
- **All tests MUST FAIL initially** (TDD requirement)

---

### T005: ✅ COMPLETE - [P] BasicRAG error handling contract test
**File**: `tests/contract/test_basic_error_handling.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-009, FR-010, FR-011, FR-012, FR-013, FR-014
**Marker**: `@pytest.mark.contract`, `@pytest.mark.error_handling`, `@pytest.mark.basic_rag`

**Implementation**:
1. Create `TestBasicRAGErrorHandling` class
2. Implement `test_missing_api_key_error_is_actionable`:
   - Mock os.environ to remove OPENAI_API_KEY
   - Execute query, expect ConfigurationError
   - Verify error message includes "OPENAI_API_KEY" and "export"
3. Implement `test_database_connection_retries_with_backoff`:
   - Mock connection to fail 2 times, succeed 3rd time
   - Execute query, verify 3 connection attempts
   - Verify retry logging in caplog
4. Implement `test_error_includes_pipeline_context`:
   - Trigger error, verify message includes "BasicRAG", "query", state info
5. Implement `test_dimension_mismatch_error_actionable`:
   - Mock embedding to wrong dimension (768)
   - Expect DimensionMismatchError with "384" and "768" in message
   - Verify fix suggestion includes "config.yaml"

**Acceptance**:
- Error messages follow Error → Context → Expected → Actual → Fix template
- All error messages include actionable guidance (FR-010)
- Tests use caplog for logging validation
- Tests use mocker for mocking
- **All tests MUST FAIL initially**

---

### T006: ✅ COMPLETE - [P] BasicRAG dimension validation contract test
**File**: `tests/contract/test_basic_dimension_validation.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-021, FR-022, FR-023, FR-024
**Marker**: `@pytest.mark.contract`, `@pytest.mark.dimension`, `@pytest.mark.basic_rag`

**Implementation**:
1. Create `TestBasicRAGDimensionValidation` class
2. Implement `test_query_embedding_is_384_dimensions`:
   - Generate query embedding via pipeline.embedding_manager
   - Assert len(embedding) == 384
3. Implement `test_dimension_validation_before_search`:
   - Mock embedding to return 768D vector
   - Execute query, expect DimensionMismatchError
   - Verify error raised BEFORE database query
4. Implement `test_dimension_mismatch_includes_both_dimensions`:
   - Mock embedding to 512D
   - Verify error message contains "384" and "512"
5. Implement `test_error_suggests_actionable_fix`:
   - Verify error message includes keywords: "verify", "config", "re-index"
6. Implement `test_document_embedding_dimension_check` (requires DB):
   - Query RAG.SourceDocuments for sample embedding
   - Verify all document embeddings are 384D

**Acceptance**:
- Dimension validation occurs at multiple points (query, document)
- Error messages include both expected and actual dimensions (FR-023)
- Error messages suggest actionable fixes (FR-024)
- **All tests MUST FAIL initially**

---

## Phase 3.3: Contract Tests - CRAG Pipeline (TDD)

### T007: ✅ COMPLETE - [P] CRAG API contract test
**File**: `tests/contract/test_crag_contract.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-001, FR-002, FR-003, FR-004
**Marker**: `@pytest.mark.contract`, `@pytest.mark.crag`

**Implementation**:
1. Create `TestCRAGContract` class
2. Implement same tests as T004 (BasicRAG API contract)
3. Add CRAG-specific test `test_query_accepts_method_parameter`:
   - Verify method="vector" works
   - Verify method="rrf" works (CRAG-specific)
   - Verify method="hybrid" works
4. Implement `test_relevance_evaluator_in_metadata`:
   - Execute query, verify metadata includes evaluator info (if used)

**Acceptance**:
- All BasicRAG API tests ported to CRAG
- CRAG-specific methods validated
- Tests use `crag_pipeline` fixture
- **All tests MUST FAIL initially**

---

### T008: ✅ COMPLETE - [P] CRAG error handling contract test
**File**: `tests/contract/test_crag_error_handling.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-009, FR-010, FR-011, FR-012, FR-013, FR-014
**Marker**: `@pytest.mark.contract`, `@pytest.mark.error_handling`, `@pytest.mark.crag`

**Implementation**:
1. Create `TestCRAGErrorHandling` class
2. Port all BasicRAG error tests (T005)
3. Add CRAG-specific test `test_relevance_evaluator_timeout_handled`:
   - Mock evaluator to timeout
   - Verify pipeline handles gracefully (logs error, continues)
4. Add `test_evaluator_api_failure_logged`:
   - Mock evaluator API to fail
   - Verify error logged at ERROR level
   - Verify query still completes (fallback to vector search)

**Acceptance**:
- All BasicRAG error tests ported
- CRAG-specific evaluator errors handled
- **All tests MUST FAIL initially**

---

### T009: ✅ COMPLETE - [P] CRAG dimension validation contract test
**File**: `tests/contract/test_crag_dimension_validation.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-021, FR-022, FR-023, FR-024
**Marker**: `@pytest.mark.contract`, `@pytest.mark.dimension`, `@pytest.mark.crag`

**Implementation**:
1. Create `TestCRAGDimensionValidation` class
2. Port all BasicRAG dimension tests (T006)
3. Same expected dimension: 384 (all-MiniLM-L6-v2)

**Acceptance**:
- All dimension validation tests ported to CRAG
- 384D validation consistent with BasicRAG
- **All tests MUST FAIL initially**

---

### T010: ✅ COMPLETE - [P] CRAG fallback mechanism contract test
**File**: `tests/contract/test_crag_fallback_mechanism.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-015, FR-017, FR-018, FR-019, FR-020
**Marker**: `@pytest.mark.contract`, `@pytest.mark.fallback`, `@pytest.mark.crag`

**Implementation**:
1. Create `TestCRAGFallback` class
2. Implement `test_fallback_retrieves_documents_successfully`:
   - Mock relevance evaluator to fail
   - Execute query, verify contexts retrieved via fallback
   - Verify metadata.fallback_used == True
3. Implement `test_fallback_activation_logged`:
   - Mock evaluator to fail
   - Verify INFO log contains "fallback activated"
   - Verify log includes primary_method and fallback_method
4. Implement `test_fallback_preserves_query_semantics`:
   - Get baseline result (primary method)
   - Trigger fallback, get fallback result
   - Verify fallback count >= baseline count * 0.5
5. Implement `test_fallback_can_be_disabled`:
   - Disable fallback via config
   - Mock evaluator to fail
   - Verify exception raised (no fallback attempted)
6. Implement `test_all_fallbacks_exhausted_handled`:
   - Mock all methods to fail
   - Verify AllFallbacksExhaustedError raised
   - Verify error message includes all attempted methods

**Acceptance**:
- Fallback tests validate CRAG-specific strategy (evaluator → vector+text fusion)
- Logging validation via caplog
- **All tests MUST FAIL initially**

---

## Phase 3.4: Contract Tests - BasicRerankRAG Pipeline (TDD)

### T011: ✅ COMPLETE - [P] BasicRerankRAG API contract test
**File**: `tests/contract/test_basic_rerank_contract.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-001, FR-002, FR-003, FR-004
**Marker**: `@pytest.mark.contract`, `@pytest.mark.basic_rerank`

**Implementation**:
1. Create `TestBasicRerankRAGContract` class
2. Port all BasicRAG API tests (T004)
3. Add BasicRerankRAG-specific test `test_rerank_method_metadata`:
   - Execute query, verify metadata includes reranking info

**Acceptance**:
- All API contract tests ported
- BasicRerankRAG-specific reranking validated
- Uses `basic_rerank_pipeline` fixture
- **All tests MUST FAIL initially**

---

### T012: ✅ COMPLETE - [P] BasicRerankRAG error handling contract test
**File**: `tests/contract/test_basic_rerank_error_handling.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-009, FR-010, FR-011, FR-012, FR-013, FR-014
**Marker**: `@pytest.mark.contract`, `@pytest.mark.error_handling`, `@pytest.mark.basic_rerank`

**Implementation**:
1. Create `TestBasicRerankRAGErrorHandling` class
2. Port all BasicRAG error tests (T005)
3. Add BasicRerankRAG-specific test `test_reranker_model_loading_error`:
   - Mock reranker model loading to fail
   - Verify clear error message with fix guidance
4. Add `test_reranker_timeout_handled`:
   - Mock reranker to timeout
   - Verify graceful handling (fallback to initial ranking)

**Acceptance**:
- All error tests ported
- Reranker-specific errors handled
- **All tests MUST FAIL initially**

---

### T013: ✅ COMPLETE - [P] BasicRerankRAG dimension validation contract test
**File**: `tests/contract/test_basic_rerank_dimension_validation.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-021, FR-022, FR-023, FR-024
**Marker**: `@pytest.mark.contract`, `@pytest.mark.dimension`, `@pytest.mark.basic_rerank`

**Implementation**:
1. Create `TestBasicRerankRAGDimensionValidation` class
2. Port all BasicRAG dimension tests (T006)
3. Same expected dimension: 384 for vector search (cross-encoder has variable dims)

**Acceptance**:
- All dimension tests ported
- 384D validation for vector search component
- **All tests MUST FAIL initially**

---

### T014: ✅ COMPLETE - [P] BasicRerankRAG fallback mechanism contract test
**File**: `tests/contract/test_basic_rerank_fallback_mechanism.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-016, FR-017, FR-018, FR-019, FR-020
**Marker**: `@pytest.mark.contract`, `@pytest.mark.fallback`, `@pytest.mark.basic_rerank`

**Implementation**:
1. Create `TestBasicRerankRAGFallback` class
2. Port fallback tests from CRAG (T010), adapt for reranker fallback
3. Fallback strategy: Cross-encoder → vector similarity ranking
4. Implement tests:
   - `test_reranker_fallback_retrieves_documents`
   - `test_reranker_fallback_activation_logged`
   - `test_reranker_fallback_preserves_semantics`
   - `test_reranker_fallback_can_be_disabled`

**Acceptance**:
- Fallback tests validate reranker → vector ranking strategy
- Metadata indicates fallback usage
- **All tests MUST FAIL initially**

---

## Phase 3.5: Contract Tests - PyLateColBERT Pipeline (TDD)

### T015: ✅ COMPLETE - [P] PyLateColBERT API contract test
**File**: `tests/contract/test_pylate_colbert_contract.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-001, FR-002, FR-003, FR-004
**Marker**: `@pytest.mark.contract`, `@pytest.mark.pylate_colbert`

**Implementation**:
1. Create `TestPyLateColBERTContract` class
2. Port all BasicRAG API tests (T004)
3. Add PyLateColBERT-specific test `test_colbert_search_metadata`:
   - Execute query, verify metadata includes ColBERT scoring info

**Acceptance**:
- All API contract tests ported
- ColBERT-specific metadata validated
- Uses `pylate_colbert_pipeline` fixture
- **All tests MUST FAIL initially**

---

### T016: ✅ COMPLETE - [P] PyLateColBERT error handling contract test
**File**: `tests/contract/test_pylate_colbert_error_handling.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-009, FR-010, FR-011, FR-012, FR-013, FR-014
**Marker**: `@pytest.mark.contract`, `@pytest.mark.error_handling`, `@pytest.mark.pylate_colbert`

**Implementation**:
1. Create `TestPyLateColBERTErrorHandling` class
2. Port all BasicRAG error tests (T005)
3. Add PyLateColBERT-specific tests:
   - `test_colbert_model_loading_error`
   - `test_colbert_score_computation_error`
   - Verify clear error messages with actionable fixes

**Acceptance**:
- All error tests ported
- ColBERT-specific errors handled
- **All tests MUST FAIL initially**

---

### T017: ✅ COMPLETE - [P] PyLateColBERT dimension validation contract test
**File**: `tests/contract/test_pylate_colbert_dimension_validation.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-021, FR-022, FR-023, FR-024
**Marker**: `@pytest.mark.contract`, `@pytest.mark.dimension`, `@pytest.mark.pylate_colbert`

**Implementation**:
1. Create `TestPyLateColBERTDimensionValidation` class
2. Note: ColBERT uses token-level embeddings (variable dimensions)
3. Implement ColBERT-specific dimension tests:
   - `test_colbert_token_embedding_structure` (validate token count, dim per token)
   - `test_fallback_dense_vector_is_384d` (when falling back to dense search)
4. Adapt BasicRAG dimension tests for ColBERT's unique structure

**Acceptance**:
- Dimension tests account for ColBERT's token-level embeddings
- Fallback to dense vector validates 384D
- **All tests MUST FAIL initially**

---

### T018: ✅ COMPLETE - [P] PyLateColBERT fallback mechanism contract test
**File**: `tests/contract/test_pylate_colbert_fallback_mechanism.py`
**Dependencies**: T001 (fixtures)
**Requirements**: FR-015, FR-017, FR-018, FR-019, FR-020
**Marker**: `@pytest.mark.contract`, `@pytest.mark.fallback`, `@pytest.mark.pylate_colbert`

**Implementation**:
1. Create `TestPyLateColBERTFallback` class
2. Fallback strategy: ColBERT late interaction → dense vector (all-MiniLM-L6-v2)
3. Implement tests:
   - `test_colbert_fallback_to_dense_vector`
   - `test_colbert_fallback_activation_logged`
   - `test_colbert_fallback_preserves_semantics`
   - `test_colbert_fallback_can_be_disabled`

**Acceptance**:
- Fallback tests validate ColBERT → dense vector strategy
- Metadata indicates ColBERT vs dense vector retrieval
- **All tests MUST FAIL initially**

---

## Phase 3.6: Integration Tests (E2E)

### T019: ✅ COMPLETE - [P] BasicRAG end-to-end integration test
**File**: `tests/integration/test_basic_rag_e2e.py`
**Dependencies**: T001 (fixtures), T002 (test data), T004-T006 (contract tests passing)
**Requirements**: FR-025, FR-026, FR-027, FR-028
**Marker**: `@pytest.mark.integration`, `@pytest.mark.requires_database`, `@pytest.mark.basic_rag`

**Implementation**:
1. Create `TestBasicRAGIntegration` class
2. Implement `test_full_query_path_with_real_db`:
   - Load sample documents from T002
   - Execute load_documents(), verify storage
   - Execute query(), verify retrieval → generation
   - Assert response quality (context_count ≥ 1)
3. Implement `test_document_loading_workflow`:
   - Load 5 documents, verify embeddings generated
   - Verify documents stored in IRIS
   - Verify document count matches
4. Implement `test_response_quality_metrics`:
   - Execute query, verify answer is non-empty
   - Verify source attribution present
   - Verify execution time logged

**Acceptance**:
- Full pipeline workflow validated (load → embed → store → retrieve → generate)
- Uses live IRIS database (Constitutional requirement III)
- Test completes in <30s
- Response quality assertions pass

---

### T020: ✅ COMPLETE - [P] CRAG end-to-end integration test
**File**: `tests/integration/test_crag_e2e.py`
**Dependencies**: T001, T002, T007-T010
**Requirements**: FR-025, FR-026, FR-027, FR-028
**Marker**: `@pytest.mark.integration`, `@pytest.mark.requires_database`, `@pytest.mark.crag`

**Implementation**:
1. Create `TestCRAGIntegration` class
2. Port all BasicRAG integration tests (T019)
3. Add CRAG-specific test `test_relevance_evaluation_in_query_path`:
   - Execute query, verify relevance evaluation occurred
   - Verify metadata includes evaluation score (if available)
4. Add `test_web_search_augmentation_when_low_relevance`:
   - Mock relevance evaluator to return low score
   - Verify web search augmentation triggered (if configured)

**Acceptance**:
- All BasicRAG integration tests ported
- CRAG-specific relevance evaluation validated
- Uses live IRIS database
- Test completes in <30s

---

### T021: ✅ COMPLETE - [P] BasicRerankRAG end-to-end integration test
**File**: `tests/integration/test_basic_rerank_e2e.py`
**Dependencies**: T001, T002, T011-T014
**Requirements**: FR-025, FR-026, FR-027, FR-028
**Marker**: `@pytest.mark.integration`, `@pytest.mark.requires_database`, `@pytest.mark.basic_rerank`

**Implementation**:
1. Create `TestBasicRerankRAGIntegration` class
2. Port all BasicRAG integration tests (T019)
3. Add BasicRerankRAG-specific test `test_reranking_in_query_path`:
   - Execute query, verify reranking occurred
   - Verify metadata includes reranking info
   - Verify contexts reordered (top context most relevant)

**Acceptance**:
- All BasicRAG integration tests ported
- Reranking validated in query path
- Uses live IRIS database
- Test completes in <30s

---

### T022: ✅ COMPLETE - [P] PyLateColBERT end-to-end integration test
**File**: `tests/integration/test_pylate_colbert_e2e.py`
**Dependencies**: T001, T002, T015-T018
**Requirements**: FR-025, FR-026, FR-027, FR-028
**Marker**: `@pytest.mark.integration`, `@pytest.mark.requires_database`, `@pytest.mark.pylate_colbert`

**Implementation**:
1. Create `TestPyLateColBERTIntegration` class
2. Port all BasicRAG integration tests (T019)
3. Add PyLateColBERT-specific test `test_colbert_late_interaction_search`:
   - Execute query, verify ColBERT search occurred
   - Verify metadata includes ColBERT scoring
   - Verify late interaction computed correctly

**Acceptance**:
- All BasicRAG integration tests ported
- ColBERT late interaction validated
- Uses live IRIS database
- Test completes in <30s

---

## Phase 3.7: Validation & Polish

### T023: ✅ COMPLETE - Run contract test suite and verify <30s execution
**File**: N/A (validation task)
**Dependencies**: T004-T018 (all contract tests)
**Requirements**: FR-005

**Implementation**:
1. Execute all contract tests: `pytest tests/contract/ -v`
2. Measure total execution time
3. Verify total time <30 seconds
4. If >30s, identify slow tests and optimize:
   - Reduce test data size
   - Mock expensive operations
   - Parallelize independent tests

**Acceptance**:
- All contract tests pass
- Total execution time <30 seconds
- CI/CD compatible performance

---

### T024: ✅ COMPLETE - Validate functional requirements coverage
**File**: `specs/036-retrofit-graphrag-s/REQUIREMENTS_TRACEABILITY.md`
**Dependencies**: All test tasks (T004-T022)
**Requirements**: All FR-001 to FR-028

**Implementation**:
1. Create traceability matrix mapping FR to tests
2. For each FR-001 to FR-028:
   - List test file validating requirement
   - List specific test method
   - Verify test passes
3. Document format:
   ```markdown
   ## FR-001: Contract tests validate core methods
   - Test: tests/contract/test_basic_rag_contract.py::test_query_method_exists
   - Status: ✅ PASS
   - Pipelines: BasicRAG, CRAG, BasicRerankRAG, PyLateColBERT
   ```

**Acceptance**:
- All 28 functional requirements have test coverage
- Traceability matrix complete
- All tests passing

---

### T025: ✅ COMPLETE - [P] Generate test coverage report
**File**: N/A (validation task)
**Dependencies**: T004-T022

**Implementation**:
1. Run pytest with coverage: `pytest --cov=iris_rag tests/`
2. Generate HTML report: `pytest --cov=iris_rag --cov-report=html tests/`
3. Verify coverage for target pipelines:
   - BasicRAG: Contract and integration tests
   - CRAG: Contract and integration tests
   - BasicRerankRAG: Contract and integration tests
   - PyLateColBERT: Contract and integration tests
4. Document coverage percentages in plan.md

**Acceptance**:
- Coverage report generated
- All 4 pipelines have test coverage
- Report accessible in htmlcov/index.html

---

### T026: ✅ COMPLETE - Update Make targets for test execution
**File**: `Makefile`
**Dependencies**: All test tasks

**Implementation**:
1. Add target `test-contract`:
   ```makefile
   .PHONY: test-contract
   test-contract: ## Run contract tests (<30s)
   	pytest tests/contract/ -v -m contract
   ```
2. Add target `test-integration-pipelines`:
   ```makefile
   .PHONY: test-integration-pipelines
   test-integration-pipelines: ## Run pipeline integration tests
   	pytest tests/integration/ -v -m "integration and (basic_rag or crag or basic_rerank or pylate_colbert)"
   ```
3. Add target `test-all-pipelines`:
   ```makefile
   .PHONY: test-all-pipelines
   test-all-pipelines: ## Run all pipeline tests
   	pytest tests/contract/ tests/integration/ -v -m "basic_rag or crag or basic_rerank or pylate_colbert"
   ```
4. Add pipeline-specific targets:
   ```makefile
   test-basic-rag: ## Test BasicRAG only
   test-crag: ## Test CRAG only
   test-basic-rerank: ## Test BasicRerankRAG only
   test-pylate-colbert: ## Test PyLateColBERT only
   ```

**Acceptance**:
- Make targets added to Makefile
- Targets documented with ## comments
- Test execution via make commands working

---

### T027: ✅ COMPLETE - [P] Update quickstart.md with execution results
**File**: `specs/036-retrofit-graphrag-s/quickstart.md`
**Dependencies**: T023, T024, T025

**Implementation**:
1. Update "Expected Output" sections with actual test results
2. Add actual performance benchmarks:
   - Contract tests: Xs (target: <30s)
   - Integration tests: Xm Ys (target: <2m)
   - Total: Xm Ys (target: <5m)
3. Update validation checklist with actual test counts:
   - BasicRAG: X tests passing
   - CRAG: X tests passing
   - BasicRerankRAG: X tests passing
   - PyLateColBERT: X tests passing
4. Add troubleshooting entries for any issues encountered

**Acceptance**:
- Quickstart.md updated with actual results
- Performance benchmarks documented
- Troubleshooting guide enhanced

---

### T028: ✅ COMPLETE - Validate against success criteria
**File**: `specs/036-retrofit-graphrag-s/VALIDATION_REPORT.md`
**Dependencies**: All tasks

**Implementation**:
1. Create validation report documenting success criteria from spec.md
2. Validate test coverage success criteria:
   - ✅ All 4 pipelines have comprehensive contract test suites
   - ✅ All 4 pipelines have integration tests
   - ✅ Contract test execution <30 seconds
3. Validate error handling success criteria:
   - ✅ All configuration errors provide diagnostic messages
   - ✅ All runtime failures trigger fallbacks with logging
   - ✅ All dimension mismatches detected with clear errors
4. Validate quality success criteria:
   - ✅ Contract tests validate API contracts
   - ✅ Integration tests assert response quality
   - ✅ All tests pass in CI/CD with mock services
5. Validate documentation success criteria:
   - ✅ Each pipeline has documented test patterns
   - ✅ Error messages reference configuration
   - ✅ Fallback mechanisms documented

**Acceptance**:
- All success criteria from spec.md validated
- Validation report complete
- All criteria passing

---

## Dependencies

**Setup (T001-T003)**:
- Must complete before all test tasks

**Contract Tests (T004-T018)**:
- Depend on: T001 (fixtures), T002 (test data), T003 (markers)
- Can run in parallel [P] - different files
- TDD requirement: MUST FAIL initially, then pass after implementation

**Integration Tests (T019-T022)**:
- Depend on: Contract tests passing (T004-T018)
- Can run in parallel [P] - different files
- Require live IRIS database

**Validation (T023-T028)**:
- Depend on: All test tasks complete
- T025, T027 can run in parallel [P]
- T028 must be last (validates all criteria)

---

## Parallel Execution Examples

### Example 1: Setup Tasks (Run in sequence)
```bash
# T001-T003 must run sequentially (T001 modifies conftest.py)
# Execute T001, then T002 and T003 in parallel
```

### Example 2: BasicRAG Contract Tests (Parallel)
```bash
# T004-T006 can run in parallel (different files)
Task: "Create tests/contract/test_basic_rag_contract.py with API contract tests"
Task: "Create tests/contract/test_basic_error_handling.py with error tests"
Task: "Create tests/contract/test_basic_dimension_validation.py with dimension tests"
```

### Example 3: All Contract Tests (Parallel by Pipeline)
```bash
# All contract tests can run in parallel (16 different files)
Task: "Create BasicRAG contract tests (T004-T006)"
Task: "Create CRAG contract tests (T007-T010)"
Task: "Create BasicRerankRAG contract tests (T011-T014)"
Task: "Create PyLateColBERT contract tests (T015-T018)"
```

### Example 4: Integration Tests (Parallel)
```bash
# T019-T022 can run in parallel (different files)
Task: "Create tests/integration/test_basic_rag_e2e.py"
Task: "Create tests/integration/test_crag_e2e.py"
Task: "Create tests/integration/test_basic_rerank_e2e.py"
Task: "Create tests/integration/test_pylate_colbert_e2e.py"
```

---

## Notes

**TDD Requirements**:
- All contract tests (T004-T018) MUST be written first
- Tests MUST FAIL initially (no implementation exists yet)
- This feature creates TEST INFRASTRUCTURE only
- Pipeline implementations may need updates to pass tests

**Performance Requirements**:
- Contract tests: <30s total (FR-005)
- Integration tests: <2m total
- Individual test: <10s recommended

**Constitutional Compliance**:
- All tests use live IRIS database (Requirement III)
- Tests validate framework patterns (Requirement I)
- Tests validate error handling (Requirement VI)

**Commit Strategy**:
- Commit after each phase (Setup, Contract Tests, Integration Tests, Validation)
- Each commit should include passing tests for that phase

---

## Task Generation Rules Applied

1. ✅ From Contracts: 4 contract files → 16 contract test tasks [P]
2. ✅ From Data Model: 8 test entities → fixture tasks (T001)
3. ✅ From User Stories: 8 acceptance scenarios → 4 integration test tasks [P]
4. ✅ Ordering: Setup → Tests → Validation
5. ✅ Dependencies: Setup blocks all, contract tests block integration

---

## Validation Checklist

- [x] All contracts have corresponding tests (4 contracts → 16 test files)
- [x] All entities have fixture tasks (8 entities → T001)
- [x] All tests come before implementation (TDD workflow)
- [x] Parallel tasks truly independent (different files)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task

---

**Total Tasks**: 28
**Estimated Execution Time**:
- Setup: 30 minutes
- Contract Tests: 4-6 hours (TDD implementation)
- Integration Tests: 2-3 hours
- Validation: 1 hour
- **Total**: 8-10 hours

**Status**: ✅ Ready for execution via `/tasks` command
