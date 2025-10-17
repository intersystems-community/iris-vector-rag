# Tasks: Testing Framework Fixes for Coverage and Functional Correctness

**Feature**: 025-fixes-for-testing
**Branch**: `025-fixes-for-testing`
**Input**: Design documents from `/specs/025-fixes-for-testing/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Execution Flow (main)
```
1. Load plan.md from feature directory ✅
   → Tech stack: Python 3.12, pytest 8.4.1, IRIS Database
   → Structure: Single project (tests/ directory)
2. Load optional design documents ✅
   → data-model.md: 5 entities (TestCase, CoverageReport, APIContract, TestFixture, IrisConnection)
   → contracts/: 5 contract files (test execution, coverage, isolation, API alignment, GraphRAG setup)
   → research.md: 5 research areas (pytest, coverage, API contracts, GraphRAG, IRIS vector)
3. Generate tasks by category ✅
   → Setup: pytest config, fixtures, dependencies
   → Tests: 5 contract tests, E2E test fixes
   → Core: Test infrastructure, API alignment
   → Integration: IRIS database integration
   → Polish: Coverage improvements
4. Apply task rules ✅
   → Contract tests marked [P] (different files)
   → API fixes marked [P] (different test files)
   → Infrastructure tasks sequential (same config files)
5. Number tasks sequentially (T001-T085) ✅
6. Generate dependency graph ✅
7. Create parallel execution examples ✅
8. Validate task completeness ✅
   → All contracts have tests: 5/5 ✅
   → All failing tests identified: 71 total ✅
   → Coverage targets defined: 60% overall, 80% critical ✅
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- Tests organized in `tests/` directory at repository root
- Unit tests: `tests/unit/`
- E2E tests: `tests/e2e/`
- Contract tests: `tests/contract/`
- Integration tests: `tests/integration/`

---

## Phase 3.1: Setup & Infrastructure (5 tasks)

### T001: ✅ Configure pytest.ini for pytest-randomly and markers
**File**: `pytest.ini`
**Description**: Add `-p no:randomly` flag to disable pytest-randomly (fixes numpy/thinc seed errors). Configure test markers (e2e, integration, requires_database, unit).
**Dependencies**: None
**Status**: COMPLETE - pytest.ini already configured with `-p no:randomly` and all required markers
**Acceptance Criteria**:
- ✅ pytest.ini contains `addopts = -p no:randomly` (line 29)
- ✅ Markers configured: e2e, integration, requires_database, unit (lines 32-45)
- ✅ Test collection works without ModuleNotFoundError

```ini
[pytest]
addopts = -p no:randomly -v
markers =
    e2e: End-to-end tests with real IRIS database
    integration: Integration tests requiring database
    requires_database: Tests that need IRIS connection
    unit: Unit tests with mocked dependencies
```

### T002: Configure .coveragerc for coverage reporting
**File**: `.coveragerc`
**Description**: Configure coverage to exclude test files, set source paths to iris_rag and common, enable HTML reports.
**Dependencies**: None
**Acceptance Criteria**:
- .coveragerc excludes `*/tests/*`, `*/test_*`, `*/__pycache__/*`
- Source includes: iris_rag, common
- HTML reports generate in htmlcov/
- Coverage precision set to 1 decimal place

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

### T003: Update tests/conftest.py with PYTHONPATH handling
**File**: `tests/conftest.py`
**Description**: Add PYTHONPATH setup to conftest.py to enable iris_rag and common imports without manual export.
**Dependencies**: None
**Acceptance Criteria**:
- Tests can import iris_rag and common modules
- No ModuleNotFoundError when running tests
- Works across different execution contexts (IDE, CLI, CI)

```python
import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
```

### T004: Create IRIS health check in conftest.py
**File**: `tests/conftest.py`
**Description**: Add pytest session hook to verify IRIS is running before test execution. Exit early with clear message if IRIS not available.
**Dependencies**: T003
**Acceptance Criteria**:
- pytest_sessionstart hook validates IRIS connection
- Clear error message if IRIS not running: "IRIS database not running. Start with: docker-compose up -d"
- Uses common/iris_port_discovery.py to find IRIS port
- Health check query succeeds: SELECT 1

```python
def pytest_sessionstart(session):
    """Verify IRIS is healthy before running tests."""
    from common.iris_port_discovery import discover_iris_port
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.connection import ConnectionManager

    port = discover_iris_port()
    if port is None:
        pytest.exit("IRIS database not running. Start with: docker-compose up -d")

    config = ConfigurationManager()
    conn_manager = ConnectionManager(config)
    conn = conn_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()

    if result[0] != 1:
        pytest.exit("IRIS database connection failed health check")
```

### T005: Update tests/e2e/conftest.py with module-scoped IRIS fixtures
**File**: `tests/e2e/conftest.py`
**Description**: Add module-scoped fixtures for IRIS connections, pipelines, and sample documents to amortize setup cost across tests.
**Dependencies**: T003, T004
**Acceptance Criteria**:
- iris_connection fixture with scope="module"
- pipeline_dependencies fixture providing config, connection, llm_func, vector_store
- sample_documents fixture providing test Document objects
- Proper teardown to close connections after module completes

```python
import pytest
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.storage.vector_store_iris import IRISVectorStore
from common.utils import get_llm_func


@pytest.fixture(scope="module")
def config_manager():
    """Module-scoped configuration manager."""
    return ConfigurationManager()


@pytest.fixture(scope="module")
def iris_connection(config_manager):
    """Module-scoped IRIS connection."""
    conn_manager = ConnectionManager(config_manager)
    conn = conn_manager.get_connection()
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def pipeline_dependencies(config_manager, iris_connection):
    """Real dependencies for E2E testing."""
    conn_manager = ConnectionManager(config_manager)
    vector_store = IRISVectorStore(conn_manager, config_manager)

    return {
        "config_manager": config_manager,
        "connection_manager": conn_manager,
        "llm_func": get_llm_func(),
        "vector_store": vector_store,
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(id="doc1", page_content="Test content 1"),
        Document(id="doc2", page_content="Test content 2"),
        Document(id="doc3", page_content="Test content 3"),
    ]
```

---

## Phase 3.2: Contract Tests (TDD) ⚠️ MUST COMPLETE BEFORE 3.3 (5 tasks, all [P])

### T006 [P]: Create test_pytest_execution_contract.py
**File**: `tests/contract/test_pytest_execution_contract.py`
**Description**: Implement contract tests validating pytest configuration, PYTHONPATH, IRIS connection, health checks, markers, and execution time.
**Dependencies**: T001, T002, T003, T004, T005
**Acceptance Criteria**:
- 6 contract tests pass (REQ-1 through REQ-6)
- Validates pytest.ini has `-p no:randomly`
- Validates PYTHONPATH enables imports
- Validates IRIS connection fixture available
- Validates IRIS health check passes
- Validates pytest markers configured
- Validates test suite runs in < 2 minutes

**Contract**: `specs/025-fixes-for-testing/contracts/test_execution_contract.md`

### T007 [P]: Create test_coverage_contract.py
**File**: `tests/contract/test_coverage_contract.py`
**Description**: Implement contract tests validating coverage configuration, test file exclusion, and coverage targets.
**Dependencies**: T002
**Acceptance Criteria**:
- Validates .coveragerc excludes test files
- Validates source includes iris_rag and common
- Validates coverage targets (60% overall, 80% critical)
- Coverage reports generate successfully

**Contract**: `specs/025-fixes-for-testing/contracts/coverage_reporting_contract.md`

### T008 [P]: Create test_isolation_contract.py
**File**: `tests/contract/test_isolation_contract.py`
**Description**: Implement contract tests validating database cleanup, fixture scoping, and test isolation.
**Dependencies**: T005
**Acceptance Criteria**:
- Validates test data is cleaned up after execution
- Validates fixture scopes are correct (module for connections, function for documents)
- Validates tests don't pollute each other's state
- Tests can run in any order without failures

**Contract**: `specs/025-fixes-for-testing/contracts/test_isolation_contract.md`

### T009 [P]: Create test_api_contract.py
**File**: `tests/contract/test_api_contract.py`
**Description**: Implement contract tests validating test expectations match production API signatures (60 known mismatches).
**Dependencies**: None (validates existing code)
**Acceptance Criteria**:
- Validates BasicRAG load_documents signature
- Validates CRAG pipeline query signature
- Validates GraphRAG entity extraction signature
- Validates PyLate pipeline signatures
- Validates vector store API signatures
- Documents all 60 API mismatches for fixing

**Contract**: `specs/025-fixes-for-testing/contracts/api_alignment_contract.md`

### T010 [P]: Create test_graphrag_contract.py
**File**: `tests/contract/test_graphrag_contract.py`
**Description**: Implement contract tests validating GraphRAG test setup, dependencies, and error handling (11 known errors).
**Dependencies**: None (validates existing code)
**Acceptance Criteria**:
- Validates GraphRAG dependencies available or gracefully skips
- Validates entity extraction setup
- Validates graph storage integration
- Validates LLM configuration for entity extraction
- Documents all 11 GraphRAG errors for fixing

**Contract**: `specs/025-fixes-for-testing/contracts/graphrag_setup_contract.md`

---

## Phase 3.3: Fix Infrastructure Issues (5 tasks)

### T011: Fix IRIS vector store TO_VECTOR datatype usage
**File**: `common/db_vector_utils.py`, `common/vector_sql_utils.py`
**Description**: Update all TO_VECTOR() calls to use DOUBLE datatype (not FLOAT) to match VECTOR(DOUBLE, dimension) schema. Embed vector strings directly in SQL (no parameter markers).
**Dependencies**: None
**Acceptance Criteria**:
- All TO_VECTOR() calls use DOUBLE datatype
- Vector strings embedded directly in SQL (not as ? parameters)
- No "Cannot perform vector operation on vectors of different datatypes" errors
- Vector insertion and search tests pass

**Files to modify**:
- `common/db_vector_utils.py:77` - Change FLOAT → DOUBLE
- `common/db_vector_utils.py:138` - Change FLOAT → DOUBLE
- `common/vector_sql_utils.py:503` - Change FLOAT → DOUBLE, embed vector string

### T012: Add database cleanup fixtures to E2E tests
**File**: `tests/e2e/conftest.py`
**Description**: Add teardown fixtures to clean up test documents from IRIS after each test to prevent state pollution.
**Dependencies**: T005
**Acceptance Criteria**:
- Fixture deletes test documents after test execution
- Uses test document ID patterns to identify cleanup candidates
- Prevents cross-test state pollution
- Tests pass when run in any order

```python
@pytest.fixture
def cleanup_test_documents(iris_connection):
    """Clean up test documents after each test."""
    yield
    # Cleanup logic after test
    cursor = iris_connection.cursor()
    # Delete documents with test IDs
    cursor.execute("DELETE FROM RAG.SourceDocuments WHERE id LIKE 'test_%' OR id LIKE 'doc%'")
    iris_connection.commit()
```

### T013: Create test result aggregation utility
**File**: `tests/utils/test_aggregator.py`
**Description**: Create utility to aggregate test results (TestCase entities) for reporting and analysis.
**Dependencies**: None
**Acceptance Criteria**:
- Parses pytest JSON output
- Creates TestCase entities with status, execution_time, coverage_lines
- Aggregates by test suite (unit, e2e, integration)
- Generates summary statistics

**Based on**: `specs/025-fixes-for-testing/data-model.md` TestCase entity

### T014: Create coverage trend tracking utility
**File**: `tests/utils/coverage_tracker.py`
**Description**: Create utility to track coverage trends over time (CoverageReport entities).
**Dependencies**: T002
**Acceptance Criteria**:
- Parses coverage reports from .coverage file
- Creates CoverageReport entities per module
- Tracks coverage_percentage, missing_lines
- Identifies modules below target (60% overall, 80% critical)

**Based on**: `specs/025-fixes-for-testing/data-model.md` CoverageReport entity

### T015: Add pytest execution time monitoring
**File**: `tests/conftest.py`
**Description**: Add pytest hooks to monitor and report execution time per test and overall suite.
**Dependencies**: T003
**Acceptance Criteria**:
- pytest_runtest_setup captures start time
- pytest_runtest_teardown calculates duration
- Warns if individual test exceeds 5 seconds
- Fails if total suite exceeds 2 minutes
- Reports slowest 10 tests

---

## Phase 3.4: Fix API Alignment Issues - Basic Pipelines (10 tasks, all [P])

### T016 [P]: Fix test_basic_pipeline_e2e.py load_documents API
**File**: `tests/e2e/test_basic_pipeline_e2e.py`
**Description**: Update all load_documents calls to pass documents as kwarg, not positional arg. Change `load_documents(documents)` to `load_documents("", documents=documents)`.
**Dependencies**: T009 (contract test identifies mismatches)
**Acceptance Criteria**:
- All 5 load_documents calls updated to use kwarg
- Tests in TestBasicRAGPipelineDocumentLoading pass
- No API mismatch errors

**Lines to fix**: 77, 88, 104, 113, 118, 121

### T017 [P]: Fix test_basic_rerank_pipeline_e2e.py load_documents API
**File**: `tests/e2e/test_basic_rerank_pipeline_e2e.py`
**Description**: Update BasicRerankPipeline load_documents calls to match API signature.
**Dependencies**: T009
**Acceptance Criteria**:
- load_documents calls use correct kwarg signature
- TestBasicRerankPipelineDocumentLoading tests pass

### T018 [P]: Fix test_configuration_e2e.py reload test
**File**: `tests/e2e/test_configuration_e2e.py`
**Description**: Fix test_configuration_reload_e2e to match actual ConfigurationManager reload behavior.
**Dependencies**: T009
**Acceptance Criteria**:
- Test validates actual reload() signature
- Configuration reload test passes (1 currently failing)

### T019 [P]: Fix test_core_framework_e2e.py API mismatches
**File**: `tests/e2e/test_core_framework_e2e.py`
**Description**: Fix 4 failing tests in core framework E2E tests (document ingestion, retrieval, RAG pipeline, connection resilience).
**Dependencies**: T009
**Acceptance Criteria**:
- test_document_ingestion_e2e passes
- test_document_retrieval_with_queries_e2e passes
- test_basic_rag_pipeline_e2e passes
- test_database_connection_resilience_e2e passes

---

## Phase 3.5: Fix API Alignment Issues - CRAG Pipeline (20 tasks, all [P])

### T020 [P]: Fix test_crag_pipeline_e2e.py - Document Loading
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_load_documents_with_embeddings to match CRAG pipeline load_documents API.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineDocumentLoading::test_load_documents_with_embeddings passes

### T021 [P]: Fix test_crag_pipeline_e2e.py - Confident Retrieval Metadata
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_confident_query_metadata to validate actual CRAG metadata structure.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineConfidentRetrieval::test_confident_query_metadata passes
- Validates actual metadata fields from CRAG pipeline

### T022 [P]: Fix test_crag_pipeline_e2e.py - Ambiguous Query Enhancement
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_ambiguous_query_enhancement to match CRAG ambiguous handling behavior.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineAmbiguousRetrieval::test_ambiguous_query_enhancement passes

### T023 [P]: Fix test_crag_pipeline_e2e.py - Ambiguous Metadata
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_ambiguous_query_metadata to validate actual ambiguous metadata structure.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineAmbiguousRetrieval::test_ambiguous_query_metadata passes

### T024 [P]: Fix test_crag_pipeline_e2e.py - Chunk Enhancement
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_ambiguous_with_chunk_enhancement to match CRAG chunk processing.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineAmbiguousRetrieval::test_ambiguous_with_chunk_enhancement passes

### T025 [P]: Fix test_crag_pipeline_e2e.py - Disoriented Expansion
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_disoriented_query_expansion to match CRAG disoriented handling.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineDisorientedRetrieval::test_disoriented_query_expansion passes

### T026 [P]: Fix test_crag_pipeline_e2e.py - Corrective Actions Confident
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_corrective_action_for_confident to validate actual corrective logic.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineCorrectiveActions::test_corrective_action_for_confident passes

### T027 [P]: Fix test_crag_pipeline_e2e.py - Corrective Actions Ambiguous
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_corrective_action_for_ambiguous to validate actual corrective logic.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineCorrectiveActions::test_corrective_action_for_ambiguous passes

### T028 [P]: Fix test_crag_pipeline_e2e.py - Evaluator High Scores
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_evaluator_with_high_scores to match retrieval evaluator API.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineRetrievalEvaluator::test_evaluator_with_high_scores passes

### T029 [P]: Fix test_crag_pipeline_e2e.py - Evaluator No Documents
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_evaluator_with_no_documents to handle empty document sets.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineRetrievalEvaluator::test_evaluator_with_no_documents passes

### T030 [P]: Fix test_crag_pipeline_e2e.py - Evaluator Status Values
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_evaluator_status_values to validate actual status enum.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineRetrievalEvaluator::test_evaluator_status_values passes

### T031 [P]: Fix test_crag_pipeline_e2e.py - Answer Generation Confident
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_answer_generation_confident to match CRAG answer generation.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineAnswerGeneration::test_answer_generation_confident passes

### T032 [P]: Fix test_crag_pipeline_e2e.py - Answer Generation Without LLM
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_answer_generation_without_llm to handle no-LLM mode.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineAnswerGeneration::test_answer_generation_without_llm passes

### T033 [P]: Fix test_crag_pipeline_e2e.py - Sequential Queries
**File**: `tests/e2e/test_crag_pipeline_e2e.py`
**Description**: Fix test_sequential_queries_different_statuses to validate multiple query handling.
**Dependencies**: T009
**Acceptance Criteria**:
- TestCRAGPipelineIntegration::test_sequential_queries_different_statuses passes

---

## Phase 3.6: Fix API Alignment Issues - GraphRAG Pipeline (40 tasks, all [P])

### T034 [P]: Fix test_graphrag_pipeline_e2e.py - Document Entity Extraction
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_load_documents_extracts_entities to match GraphRAG entity extraction API.
**Dependencies**: T010 (GraphRAG contract)
**Acceptance Criteria**:
- TestGraphRAGPipelineDocumentLoading::test_load_documents_extracts_entities passes or skips with explanation

### T035 [P]: Fix test_graphrag_pipeline_e2e.py - Document Relationship Extraction
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_load_documents_extracts_relationships to match relationship extraction.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineDocumentLoading::test_load_documents_extracts_relationships passes or skips

### T036 [P]: Fix test_graphrag_pipeline_e2e.py - Single Document Entities
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_load_single_document_with_entities for single document processing.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineDocumentLoading::test_load_single_document_with_entities passes or skips

### T037 [P]: Fix test_graphrag_pipeline_e2e.py - Entity Types
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_entity_types_extracted to validate actual entity type enum.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineEntityExtraction::test_entity_types_extracted passes or skips

### T038 [P]: Fix test_graphrag_pipeline_e2e.py - Entity Extraction from Text
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_entity_extraction_from_text to match extraction API.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineEntityExtraction::test_entity_extraction_from_text passes or skips

### T039 [P]: Fix test_graphrag_pipeline_e2e.py - Multiple Documents Extraction
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_multiple_documents_entity_extraction for batch processing.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineEntityExtraction::test_multiple_documents_entity_extraction passes or skips

### T040 [P]: Fix test_graphrag_pipeline_e2e.py - Relationship Storage
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_relationship_storage to validate graph storage.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineRelationshipStorage::test_relationship_storage passes or skips

### T041 [P]: Fix test_graphrag_pipeline_e2e.py - Bidirectional Relationships
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_bidirectional_relationships for graph relationship logic.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineRelationshipStorage::test_bidirectional_relationships passes or skips

### T042 [P]: Fix test_graphrag_pipeline_e2e.py - Validation with Graph
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_validation_with_populated_graph for graph validation.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineValidation::test_validation_with_populated_graph passes or skips

### T043 [P]: Fix test_graphrag_pipeline_e2e.py - Fallback to Vector
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_fallback_to_vector_search for GraphRAG fallback mode.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineFallback::test_fallback_to_vector_search passes or skips

### T044 [P]: Fix test_graphrag_pipeline_e2e.py - Fallback Metadata
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_fallback_metadata to validate fallback metadata structure.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineFallback::test_fallback_metadata passes or skips

### T045 [P]: Fix test_graphrag_pipeline_e2e.py - Invalid top_k
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_query_with_invalid_top_k for edge case handling.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineErrorHandling::test_query_with_invalid_top_k passes or skips

### T046 [P]: Fix test_graphrag_pipeline_e2e.py - Query Error Recovery
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_query_error_recovery for error handling validation.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineErrorHandling::test_query_error_recovery passes or skips

### T047 [P]: Fix test_graphrag_pipeline_e2e.py - Complete Workflow
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_complete_graphrag_workflow for end-to-end workflow.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineIntegration::test_complete_graphrag_workflow passes or skips

### T048 [P]: Fix test_graphrag_pipeline_e2e.py - Large Knowledge Graph
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_large_knowledge_graph for scale testing.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineIntegration::test_large_knowledge_graph passes or skips

### T049 [P]: Fix test_graphrag_pipeline_e2e.py - Sequential Graph Queries
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_sequential_queries_on_graph for multiple query handling.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineIntegration::test_sequential_queries_on_graph passes or skips

### T050 [P]: Fix test_graphrag_pipeline_e2e.py - Execution Time
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_execution_time_tracking for performance monitoring.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelinePerformance::test_execution_time_tracking passes or skips

### T051 [P]: Fix test_graphrag_pipeline_e2e.py - Graph Traversal Performance
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_graph_traversal_performance for traversal timing.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelinePerformance::test_graph_traversal_performance passes or skips

### T052 [P]: Fix test_graphrag_pipeline_e2e.py - Entity Extraction Performance
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix test_entity_extraction_performance for extraction timing.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelinePerformance::test_entity_extraction_performance passes or skips

### T053 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Simple Graph Query
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_simple_graph_query (setup failure). Either fix GraphRAG setup or skip with explanation.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineQuerying::test_simple_graph_query passes or skips (no ERROR)

### T054 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Entity Matching
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_query_with_entity_matching.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineQuerying::test_query_with_entity_matching passes or skips (no ERROR)

### T055 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Answer Generation
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_query_with_answer_generation.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineQuerying::test_query_with_answer_generation passes or skips (no ERROR)

### T056 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Related Entities
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_graph_traversal_finds_related_entities.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineGraphTraversal::test_graph_traversal_finds_related_entities passes or skips (no ERROR)

### T057 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Multi-hop Traversal
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_multi_hop_traversal.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineGraphTraversal::test_multi_hop_traversal passes or skips (no ERROR)

### T058 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Traversal Depth Limit
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_traversal_depth_limit.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineGraphTraversal::test_traversal_depth_limit passes or skips (no ERROR)

### T059 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Find Seed Entities
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_find_seed_entities_for_query.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineSeedEntityFinding::test_find_seed_entities_for_query passes or skips (no ERROR)

### T060 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Seed Entities Relevance
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_seed_entities_relevance.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineSeedEntityFinding::test_seed_entities_relevance passes or skips (no ERROR)

### T061 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Seed No Matches
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_seed_entities_with_no_matches.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineSeedEntityFinding::test_seed_entities_with_no_matches passes or skips (no ERROR)

### T062 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Document Retrieval
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_retrieve_documents_from_entities.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineDocumentRetrieval::test_retrieve_documents_from_entities passes or skips (no ERROR)

### T063 [P]: Resolve test_graphrag_pipeline_e2e.py - ERROR: Retrieval Metadata
**File**: `tests/e2e/test_graphrag_pipeline_e2e.py`
**Description**: Fix ERROR in test_document_retrieval_metadata.
**Dependencies**: T010
**Acceptance Criteria**:
- TestGraphRAGPipelineDocumentRetrieval::test_document_retrieval_metadata passes or skips (no ERROR)

---

## Phase 3.7: Fix API Alignment Issues - PyLate Pipeline (7 tasks, all [P])

### T064 [P]: Fix test_pylate_pipeline_e2e.py - Document Loading
**File**: `tests/e2e/test_pylate_pipeline_e2e.py`
**Description**: Fix test_document_loading_e2e to match PyLate load_documents API (fallback mode).
**Dependencies**: T009
**Acceptance Criteria**:
- TestPyLateColBERTPipelineE2E::test_document_loading_e2e passes

### T065 [P]: Fix test_pylate_pipeline_e2e.py - Query Execution
**File**: `tests/e2e/test_pylate_pipeline_e2e.py`
**Description**: Fix test_query_execution_e2e to match PyLate query API.
**Dependencies**: T009
**Acceptance Criteria**:
- TestPyLateColBERTPipelineE2E::test_query_execution_e2e passes

### T066 [P]: Fix test_pylate_pipeline_e2e.py - Multiple Queries
**File**: `tests/e2e/test_pylate_pipeline_e2e.py`
**Description**: Fix test_multiple_queries_e2e for sequential query handling.
**Dependencies**: T009
**Acceptance Criteria**:
- TestPyLateColBERTPipelineE2E::test_multiple_queries_e2e passes

### T067 [P]: Fix test_pylate_pipeline_e2e.py - Custom Parameters
**File**: `tests/e2e/test_pylate_pipeline_e2e.py`
**Description**: Fix test_query_with_custom_parameters_e2e for parameter passing.
**Dependencies**: T009
**Acceptance Criteria**:
- TestPyLateColBERTPipelineE2E::test_query_with_custom_parameters_e2e passes

### T068 [P]: Fix test_pylate_pipeline_e2e.py - Stats Tracking
**File**: `tests/e2e/test_pylate_pipeline_e2e.py`
**Description**: Fix test_stats_tracking_e2e to match PyLate stats API.
**Dependencies**: T009
**Acceptance Criteria**:
- TestPyLateColBERTPipelineE2E::test_stats_tracking_e2e passes

### T069 [P]: Fix test_pylate_pipeline_e2e.py - Metadata Preservation
**File**: `tests/e2e/test_pylate_pipeline_e2e.py`
**Description**: Fix test_document_metadata_preservation_e2e for metadata handling.
**Dependencies**: T009
**Acceptance Criteria**:
- TestPyLateColBERTPipelineE2E::test_document_metadata_preservation_e2e passes

### T070 [P]: Fix test_pylate_pipeline_e2e.py - Contexts Match
**File**: `tests/e2e/test_pylate_pipeline_e2e.py`
**Description**: Fix test_contexts_match_documents_e2e for context validation.
**Dependencies**: T009
**Acceptance Criteria**:
- TestPyLateColBERTPipelineE2E::test_contexts_match_documents_e2e passes

---

## Phase 3.8: Fix API Alignment Issues - Vector Store (8 tasks, all [P])

### T071 [P]: Fix test_vector_store_comprehensive_e2e.py - Filter by Category
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_filter_by_category to match IRIS vector store filter API.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreMetadataFiltering::test_filter_by_category passes

### T072 [P]: Fix test_vector_store_comprehensive_e2e.py - Filter by Year
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_filter_by_year for metadata filtering.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreMetadataFiltering::test_filter_by_year passes

### T073 [P]: Fix test_vector_store_comprehensive_e2e.py - Multiple Criteria
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_filter_by_multiple_criteria for complex filtering.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreMetadataFiltering::test_filter_by_multiple_criteria passes

### T074 [P]: Fix test_vector_store_comprehensive_e2e.py - No Matches
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_filter_with_no_matches for empty result handling.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreMetadataFiltering::test_filter_with_no_matches passes

### T075 [P]: Fix test_vector_store_comprehensive_e2e.py - High Similarity
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_high_similarity_threshold for threshold parameter.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreSimilarityThresholds::test_high_similarity_threshold passes

### T076 [P]: Fix test_vector_store_comprehensive_e2e.py - Low Similarity
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_low_similarity_threshold for threshold handling.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreSimilarityThresholds::test_low_similarity_threshold passes

### T077 [P]: Fix test_vector_store_comprehensive_e2e.py - Similarity Scores
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_similarity_scores_returned for score validation.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreSimilarityThresholds::test_similarity_scores_returned passes

### T078 [P]: Fix test_vector_store_comprehensive_e2e.py - Mixed Operations
**File**: `tests/e2e/test_vector_store_comprehensive_e2e.py`
**Description**: Fix test_mixed_operations for combined operations.
**Dependencies**: T009, T011
**Acceptance Criteria**:
- TestVectorStoreIntegration::test_mixed_operations passes

---

## Phase 3.9: Fix IRIS Vector Store E2E Tests (6 tasks, all [P])

### T079 [P]: Fix test_vector_store_iris_e2e.py - Document Storage
**File**: `tests/e2e/test_vector_store_iris_e2e.py`
**Description**: Fix test_document_storage_with_embeddings_e2e to validate TO_VECTOR(DOUBLE) usage.
**Dependencies**: T011
**Acceptance Criteria**:
- TestVectorStoreIRISCore::test_document_storage_with_embeddings_e2e passes
- Validates DOUBLE datatype usage

### T080 [P]: Fix test_vector_store_iris_e2e.py - Similarity Search
**File**: `tests/e2e/test_vector_store_iris_e2e.py`
**Description**: Fix test_vector_similarity_search_e2e for vector search validation.
**Dependencies**: T011
**Acceptance Criteria**:
- TestVectorStoreIRISCore::test_vector_similarity_search_e2e passes

### T081 [P]: Fix test_vector_store_iris_e2e.py - Search with Filters
**File**: `tests/e2e/test_vector_store_iris_e2e.py`
**Description**: Fix test_vector_search_with_filters_e2e for filtered search.
**Dependencies**: T011
**Acceptance Criteria**:
- TestVectorStoreIRISCore::test_vector_search_with_filters_e2e passes

### T082 [P]: Fix test_vector_store_iris_e2e.py - Large Scale Search
**File**: `tests/e2e/test_vector_store_iris_e2e.py`
**Description**: Fix test_large_scale_vector_search_e2e for performance validation.
**Dependencies**: T011
**Acceptance Criteria**:
- TestVectorStoreIRISPerformance::test_large_scale_vector_search_e2e passes

### T083 [P]: Fix test_vector_store_iris_e2e.py - HNSW Efficiency
**File**: `tests/e2e/test_vector_store_iris_e2e.py`
**Description**: Fix test_hnsw_index_efficiency_e2e for index performance.
**Dependencies**: T011
**Acceptance Criteria**:
- TestVectorStoreIRISPerformance::test_hnsw_index_efficiency_e2e passes

### T084 [P]: Fix test_vector_store_iris_e2e.py - Empty Search
**File**: `tests/e2e/test_vector_store_iris_e2e.py`
**Description**: Fix test_empty_search_handling_e2e and test_document_count_accuracy_e2e.
**Dependencies**: T011
**Acceptance Criteria**:
- TestVectorStoreIRISErrorHandling::test_empty_search_handling_e2e passes
- TestVectorStoreIRISErrorHandling::test_document_count_accuracy_e2e passes

---

## Phase 3.10: Coverage Improvements (1 task)

### T085: Identify and add tests for modules <60% coverage
**Files**: New test files in `tests/unit/` and `tests/e2e/`
**Description**: Run coverage report, identify modules below 60% coverage, create new unit/E2E tests to reach targets (60% overall, 80% critical).
**Dependencies**: T002, T007, All previous tasks complete
**Acceptance Criteria**:
- Coverage report generated: `pytest --cov=iris_rag --cov=common --cov-report=html`
- Modules below 60% identified
- New tests created for low-coverage modules
- Overall coverage >= 60%
- Critical modules (pipelines, storage, validation) >= 80%
- Coverage trend tracked via T014 utility

---

## Dependency Graph

```
Setup (T001-T005)
  ↓
Contract Tests [P] (T006-T010) ← Must pass before implementation fixes
  ↓
Infrastructure Fixes (T011-T015)
  ↓
API Alignment Fixes [P] (T016-T084)
  ├── Basic Pipelines [P] (T016-T019)
  ├── CRAG Pipeline [P] (T020-T033)
  ├── GraphRAG Pipeline [P] (T034-T063)
  ├── PyLate Pipeline [P] (T064-T070)
  ├── Vector Store [P] (T071-T078)
  └── IRIS Vector Store [P] (T079-T084)
  ↓
Coverage Improvements (T085)
```

## Parallel Execution Examples

### Example 1: Run all contract tests in parallel
```bash
# All contract tests can run in parallel (different files)
pytest tests/contract/ -n auto
```

### Example 2: Run API alignment fixes by pipeline in parallel
```bash
# Fix all pipeline test files concurrently
pytest tests/e2e/test_basic_pipeline_e2e.py tests/e2e/test_crag_pipeline_e2e.py tests/e2e/test_graphrag_pipeline_e2e.py tests/e2e/test_pylate_pipeline_e2e.py -n 4
```

### Example 3: Run vector store fixes in parallel
```bash
# Fix all vector store test files concurrently
pytest tests/e2e/test_vector_store_comprehensive_e2e.py tests/e2e/test_vector_store_iris_e2e.py -n 2
```

## Task Summary

**Total Tasks**: 85
- **Setup & Infrastructure**: 15 tasks (T001-T015)
- **Contract Tests [P]**: 5 tasks (T006-T010)
- **API Alignment Fixes [P]**: 64 tasks (T016-T084)
  - Basic Pipelines: 4 tasks (T016-T019)
  - CRAG Pipeline: 14 tasks (T020-T033)
  - GraphRAG Pipeline: 30 tasks (T034-T063)
  - PyLate Pipeline: 7 tasks (T064-T070)
  - Vector Store: 8 tasks (T071-T078)
  - IRIS Vector Store: 6 tasks (T079-T084)
- **Coverage Improvements**: 1 task (T085)

**Parallel Tasks**: 69 tasks can run in parallel (marked [P])
**Sequential Tasks**: 16 tasks must run sequentially

**Estimated Completion**:
- Sequential tasks: ~3-4 hours
- Parallel tasks (with 4 cores): ~2-3 hours
- **Total**: ~5-7 hours of focused work

## Validation Checklist

After completing all tasks:

- [ ] All contract tests pass (T006-T010)
- [ ] pytest.ini configured correctly (T001)
- [ ] .coveragerc configured correctly (T002)
- [ ] IRIS connection fixtures working (T005)
- [ ] IRIS health check passes (T004)
- [ ] All 71 broken tests fixed or properly skipped (T016-T084)
  - [ ] 60 failing tests → passing
  - [ ] 11 GraphRAG errors → passing or skipped with explanation
- [ ] Coverage >= 60% overall (T085)
- [ ] Critical modules >= 80% coverage (T085)
- [ ] Test suite runs in < 2 minutes (T015)
- [ ] All tests can run with `-p no:randomly` (T001)

## Success Criteria

✅ **Definition of Done**:
1. All 85 tasks completed
2. All contract tests pass
3. 0 test errors (GraphRAG tests either pass or skip)
4. ≤ 5 intentional skips (with clear reasons)
5. Coverage reports show 60%+ overall, 80%+ critical
6. Test suite executes in < 2 minutes
7. Quickstart.md validated (all commands work)
8. No pytest-randomly errors
9. IRIS vector store uses TO_VECTOR(DOUBLE) correctly
10. All API contracts validated

---

*Generated from specs/025-fixes-for-testing/ design documents*
*Based on Constitution v1.6.0 - Test-Driven Development principles*
