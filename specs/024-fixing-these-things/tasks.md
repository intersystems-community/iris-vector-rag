# Tasks: RAG-Templates Quality Improvement Initiative

**Input**: Design documents from `/specs/024-fixing-these-things/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extracted: Python 3.11+, pytest, coverage.py, Docker/IRIS
   → Structure: Single project (iris_rag/, tests/, common/)
2. Load optional design documents:
   → data-model.md: TestResult, CoverageMetric, QualityGate, TestSession entities
   → contracts/: coverage-api.yaml, quality-gate-api.yaml
   → research.md: Test repair patterns, CI/CD strategy, Docker setup
3. Generate tasks by category:
   → Setup: Docker, dependencies, test environment
   → Tests: Fix failing tests by category, add coverage tests
   → Core: Coverage framework enhancements, contract tests
   → Integration: CI/CD pipeline, quality gates
   → Polish: Documentation, performance validation
4. Apply task rules:
   → Different test files = mark [P] for parallel
   → Coverage by module = mark [P]
   → CI/CD config files = mark [P]
5. Number tasks sequentially (T001-T040)
6. Generate dependency graph
7. SUCCESS: 40 tasks ready for execution
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
Single project structure:
- **Source**: `iris_rag/`, `common/`
- **Tests**: `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/contract/`
- **CI/CD**: `.github/workflows/`
- **Specs**: `specs/024-fixing-these-things/`

---

## Phase 3.1: Environment Setup

- [ ] **T001** Create Docker test environment configuration in `docker-compose.test.yml` with IRIS database
- [ ] **T002** Update `pytest.ini` to disable pytest-randomly and configure coverage settings
- [ ] **T003** [P] Create test fixtures in `tests/conftest.py` for IRIS database connection and cleanup
- [ ] **T004** [P] Create test environment documentation in `docs/TEST_SETUP.md` (Docker setup, Python deps, env vars)
- [ ] **T005** Verify IRIS database connectivity with health check script

---

## Phase 3.2: Test Infrastructure Fixes

- [ ] **T006** Fix shared test fixtures in `tests/conftest.py` to properly initialize ConfigurationManager
- [ ] **T007** [P] Update `tests/unit/conftest.py` with proper mock configurations for critical modules
- [ ] **T008** [P] Update `tests/integration/conftest.py` with IRIS database fixtures and cleanup
- [ ] **T009** Create baseline performance benchmark file in `tests/benchmarks/baseline.json`

---

## Phase 3.3: Test Repair - AttributeError Fixes (40% of failures)

- [ ] **T010** [P] Fix AttributeError in `tests/unit/test_config_unit.py` - align with ConfigurationManager API
- [ ] **T011** [P] Fix AttributeError in `tests/unit/test_validation_unit.py` - update validator method calls
- [ ] **T012** [P] Fix AttributeError in `tests/unit/test_pipelines_unit.py` - align with pipeline interfaces
- [ ] **T013** [P] Fix AttributeError in `tests/unit/test_services_unit.py` - update service method signatures
- [ ] **T014** [P] Fix AttributeError in `tests/unit/test_storage_unit.py` - fix VectorStore and SchemaManager methods
- [ ] **T015** [P] Fix AttributeError in `tests/unit/test_comprehensive_coverage.py` - align specialized config methods
- [ ] **T016** [P] Fix AttributeError in `tests/unit/test_strategic_coverage.py` - update pipeline and storage interfaces
- [ ] **T017** [P] Fix AttributeError in `tests/unit/test_verified_coverage.py` - fix entity storage adapter calls

---

## Phase 3.4: Test Repair - Mock Configuration Fixes (30% of failures)

- [ ] **T018** [P] Fix mock configuration in `tests/unit/test_config_unit.py` - ConfigurationManager mocks
- [ ] **T019** [P] Fix mock configuration in `tests/unit/test_services_unit.py` - EmbeddingManager and StorageService mocks
- [ ] **T020** [P] Fix mock configuration in `tests/unit/test_comprehensive_coverage.py` - VectorStore mock subscriptability
- [ ] **T021** [P] Fix mock configuration in `tests/unit/test_massive_pipeline_coverage.py` - pipeline initialization mocks
- [ ] **T022** [P] Fix mock configuration in `tests/unit/test_strategic_coverage.py` - database operation mocks

---

## Phase 3.5: Test Repair - Setup and Type Fixes (30% of failures)

- [ ] **T023** [P] Fix missing config setup in `tests/unit/test_config_unit.py` - provide required database config
- [ ] **T024** [P] Fix type mismatches in `tests/unit/test_storage_unit.py` - correct return type expectations
- [ ] **T025** [P] Fix validation errors in `tests/unit/test_validation_unit.py` - requirements structure validation
- [ ] **T026** [P] Fix import errors in `tests/unit/test_final_massive_coverage.py` - SentenceTransformer path

---

## Phase 3.6: Coverage Enhancement - Config Module (Target: 80%)

- [ ] **T027** [P] Add unit tests for ConfigurationManager.get() in `tests/unit/test_config_coverage_get.py`
- [ ] **T028** [P] Add unit tests for ConfigurationManager.set() in `tests/unit/test_config_coverage_set.py`
- [ ] **T029** [P] Add unit tests for config validation in `tests/unit/test_config_coverage_validation.py`
- [ ] **T030** [P] Add unit tests for PipelineConfigService in `tests/unit/test_pipeline_config_service.py`

---

## Phase 3.7: Coverage Enhancement - Validation Module (Target: 80%)

- [ ] **T031** [P] Add unit tests for PipelineRequirements in `tests/unit/test_requirements_coverage.py`
- [ ] **T032** [P] Add unit tests for ValidationOrchestrator in `tests/unit/test_orchestrator_coverage.py`
- [ ] **T033** [P] Add unit tests for ValidatedPipelineFactory in `tests/unit/test_factory_coverage.py`

---

## Phase 3.8: Coverage Enhancement - Pipelines Module (Target: 80%)

- [ ] **T034** [P] Add unit tests for BasicRAGPipeline in `tests/unit/test_basic_pipeline_coverage.py`
- [ ] **T035** [P] Add unit tests for CRAGPipeline in `tests/unit/test_crag_pipeline_coverage.py`
- [ ] **T036** [P] Add unit tests for GraphRAGPipeline in `tests/unit/test_graphrag_pipeline_coverage.py`

---

## Phase 3.9: Coverage Enhancement - Services Module (Target: 80%)

- [ ] **T037** [P] Add unit tests for EntityExtractionService in `tests/unit/test_entity_extraction_coverage.py`
- [ ] **T038** [P] Add unit tests for StorageService in `tests/unit/test_storage_service_coverage.py`

---

## Phase 3.10: Coverage Enhancement - Storage Module (Target: 80%)

- [ ] **T039** [P] Add unit tests for IRISVectorStore in `tests/unit/test_vector_store_coverage.py`
- [ ] **T040** [P] Add unit tests for SchemaManager in `tests/unit/test_schema_manager_coverage.py`
- [ ] **T041** [P] Add unit tests for EnterpriseStorage in `tests/unit/test_enterprise_storage_coverage.py`

---

## Phase 3.11: Contract Tests for API Stability

- [ ] **T042** [P] Create contract test for coverage analyzer API in `tests/contract/test_coverage_analyzer_contract.py`
- [ ] **T043** [P] Create contract test for coverage reporter API in `tests/contract/test_coverage_reporter_contract.py`
- [ ] **T044** [P] Create contract test for coverage validator API in `tests/contract/test_coverage_validator_contract.py`
- [ ] **T045** [P] Create contract test for pipeline factory API in `tests/contract/test_pipeline_factory_contract.py`

---

## Phase 3.12: CI/CD Integration

- [ ] **T046** [P] Create GitHub Actions workflow in `.github/workflows/ci.yml` for test and lint checks
- [ ] **T047** [P] Create GitHub Actions workflow in `.github/workflows/coverage.yml` for coverage reporting
- [ ] **T048** [P] Create quality gate configuration in `.github/workflows/quality-gates.yml`
- [ ] **T049** Configure coverage thresholds in `.coveragerc` (60% overall, 80% critical modules)
- [ ] **T050** Add pre-commit hooks configuration in `.pre-commit-config.yaml`

---

## Phase 3.13: Performance and Documentation

- [ ] **T051** [P] Create performance benchmark tests in `tests/benchmarks/test_performance.py`
- [ ] **T052** [P] Update `README.md` with test setup and coverage badge
- [ ] **T053** [P] Create contributing guide in `CONTRIBUTING.md` with test requirements
- [ ] **T054** Run full test suite and generate final coverage report
- [ ] **T055** Validate all quality gates pass using `quickstart.md` validation steps

---

## Dependencies

**Sequential Dependencies**:
- T001 (Docker setup) → T003, T005, T008 (IRIS-dependent fixtures/tests)
- T002 (pytest config) → all test tasks
- T006-T009 (Test infrastructure) → all test repair and coverage tasks
- T010-T026 (Test repairs) → T054 (full validation)
- T027-T041 (Coverage enhancement) → T054 (full validation)
- T042-T045 (Contract tests) → T054 (full validation)
- T046-T050 (CI/CD) → T055 (quality gate validation)
- T054 (full test run) → T055 (final validation)

**Parallel Groups**:
- **Group 1 (Setup)**: T003, T004 can run in parallel
- **Group 2 (Test Infrastructure)**: T007, T008 can run in parallel
- **Group 3 (AttributeError Fixes)**: T010-T017 can all run in parallel
- **Group 4 (Mock Fixes)**: T018-T022 can all run in parallel
- **Group 5 (Setup/Type Fixes)**: T023-T026 can all run in parallel
- **Group 6 (Config Coverage)**: T027-T030 can all run in parallel
- **Group 7 (Validation Coverage)**: T031-T033 can all run in parallel
- **Group 8 (Pipeline Coverage)**: T034-T036 can all run in parallel
- **Group 9 (Services Coverage)**: T037-T038 can run in parallel
- **Group 10 (Storage Coverage)**: T039-T041 can all run in parallel
- **Group 11 (Contract Tests)**: T042-T045 can all run in parallel
- **Group 12 (CI/CD)**: T046-T048, T050 can run in parallel
- **Group 13 (Documentation)**: T051-T053 can run in parallel

---

## Parallel Execution Example

```bash
# After completing setup (T001-T009), launch AttributeError fixes in parallel:
# Group 3 - AttributeError Fixes (8 tasks in parallel):
Task: "Fix AttributeError in tests/unit/test_config_unit.py"
Task: "Fix AttributeError in tests/unit/test_validation_unit.py"
Task: "Fix AttributeError in tests/unit/test_pipelines_unit.py"
Task: "Fix AttributeError in tests/unit/test_services_unit.py"
Task: "Fix AttributeError in tests/unit/test_storage_unit.py"
Task: "Fix AttributeError in tests/unit/test_comprehensive_coverage.py"
Task: "Fix AttributeError in tests/unit/test_strategic_coverage.py"
Task: "Fix AttributeError in tests/unit/test_verified_coverage.py"

# After AttributeError fixes, launch Mock fixes in parallel:
# Group 4 - Mock Configuration Fixes (5 tasks in parallel):
Task: "Fix mock configuration in tests/unit/test_config_unit.py"
Task: "Fix mock configuration in tests/unit/test_services_unit.py"
Task: "Fix mock configuration in tests/unit/test_comprehensive_coverage.py"
Task: "Fix mock configuration in tests/unit/test_massive_pipeline_coverage.py"
Task: "Fix mock configuration in tests/unit/test_strategic_coverage.py"

# After all test repairs, launch coverage enhancements by module:
# Group 6-10 - Coverage Enhancement (15 tasks across 5 modules):
Task: "Add unit tests for ConfigurationManager.get() in tests/unit/test_config_coverage_get.py"
Task: "Add unit tests for ConfigurationManager.set() in tests/unit/test_config_coverage_set.py"
# ... (continue with all coverage tasks)

# Final CI/CD setup in parallel:
# Group 12 - CI/CD Configuration (4 tasks):
Task: "Create GitHub Actions workflow in .github/workflows/ci.yml"
Task: "Create GitHub Actions workflow in .github/workflows/coverage.yml"
Task: "Create quality gate configuration in .github/workflows/quality-gates.yml"
Task: "Add pre-commit hooks configuration in .pre-commit-config.yaml"
```

---

## Notes

- **[P] tasks**: Different files, no shared dependencies
- **Test-first approach**: Fix existing tests before adding new coverage
- **Systematic repair**: Group by failure type for efficient fixes
- **Critical modules first**: Focus on config, validation, pipelines, services, storage
- **Verify each group**: Run tests after completing each parallel group
- **Commit frequently**: After each task or parallel group completion
- **Docker requirement**: IRIS database must be running for integration tests

---

## Validation Checklist
*Verified during task generation*

- [x] All contracts have corresponding tests (T042-T045)
- [x] All entities (TestResult, CoverageMetric, QualityGate, TestSession) covered in implementation
- [x] Test repairs before new coverage (T010-T026 before T027-T041)
- [x] Parallel tasks truly independent (different files, no shared state)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Setup before tests (T001-T009 before all others)
- [x] CI/CD after implementation (T046-T050 after test work)
- [x] Documentation and validation at end (T051-T055)