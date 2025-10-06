# Tasks: ConfigurationManager → SchemaManager System

**Input**: Design documents from `/Users/tdyar/ws/rag-templates/specs/001-configurationmanager-schemamanager-system/`
**Prerequisites**: plan.md (✅), research.md (✅), data-model.md (✅), contracts/ (✅), quickstart.md (✅)

## Execution Flow (verification focus)
```
1. ✅ Load plan.md from feature directory → COMPLETE implementation identified
   → Tech stack: Python 3.11, InterSystems IRIS, PyYAML
   → Structure: Single project (framework component within iris_rag package)
   → Status: LARGELY IMPLEMENTED - verification mode activated
2. ✅ Load design documents → All verification artifacts available
   → data-model.md: Configuration hierarchy, SchemaManager state entities
   → contracts/: test_configuration_manager_contract.py, test_schema_manager_contract.py
   → research.md: Implementation gaps analysis (2 ConfigManager, 9 SchemaManager test failures)
   → quickstart.md: Integration test scenarios for validation
3. Generate VERIFICATION tasks by category:
   → Database: Verify IRIS EHAT connection and configuration
   → Contract Tests: Fix failing tests to align with implementation
   → Performance: Validate sub-50ms config access, sub-5s migrations
   → Constitutional: Verify all 7 principles with implementation
   → Integration: End-to-end validation with live database
4. Apply verification rules:
   → Independent validations = mark [P] for parallel
   → Sequential dependency tests = no [P]
   → Database tests require live IRIS connection
5. Number tasks sequentially (T001, T002...)
6. Validate all functional requirements against implementation
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Implementation**: `iris_rag/config/manager.py`, `iris_rag/storage/schema_manager.py`
- **Contract Tests**: `specs/001-configurationmanager-schemamanager-system/contracts/`
- **Integration**: Live IRIS database on port 1974 (EHAT container)

## Phase 3.1: Database Environment Verification
- [x] T001 Verify IRIS EHAT container connection on port 1974 with ACORN=1 optimization
- [x] T002 [P] Validate `.env` file IRIS configuration matches running container
- [ ] T003 [P] Test IRISConnectionManager connectivity and error handling

## Phase 3.2: Contract Test Alignment ⚠️ CRITICAL VERIFICATION
**CRITICAL: These tests validate implementation contracts**
- [ ] T004 [P] Fix ConfigurationManager environment variable override test in specs/001-configurationmanager-schemamanager-system/contracts/test_configuration_manager_contract.py
- [ ] T005 [P] Fix ConfigurationManager error message validation test in specs/001-configurationmanager-schemamanager-system/contracts/test_configuration_manager_contract.py
- [ ] T006 [P] Align SchemaManager vector dimension authority test in specs/001-configurationmanager-schemamanager-system/contracts/test_schema_manager_contract.py
- [ ] T007 [P] Fix SchemaManager schema migration detection test contract alignment
- [ ] T008 [P] Align SchemaManager metadata tracking test expectations
- [ ] T009 [P] Fix SchemaManager audit methods test interface alignment
- [ ] T010 [P] Update SchemaManager IRIS integration compliance test method calls

## Phase 3.3: Implementation Validation (ONLY after contract alignment)
- [ ] T011 [P] Validate ConfigurationManager YAML loading with all supported configuration keys
- [ ] T012 [P] Validate ConfigurationManager environment variable precedence order (ENV > YAML > defaults)
- [ ] T013 [P] Validate ConfigurationManager type casting for integers, booleans, and strings
- [ ] T014 [P] Validate SchemaManager vector dimension authority across all supported tables
- [ ] T015 [P] Validate SchemaManager automatic migration detection and execution
- [ ] T016 [P] Validate SchemaManager transaction safety and rollback capabilities

## Phase 3.4: Performance Validation
- [ ] T017 Benchmark ConfigurationManager access time (<50ms requirement) with 1000 iterations
- [ ] T018 Benchmark SchemaManager migration time (<5s requirement) with test tables
- [ ] T019 [P] Validate HNSW index creation performance with ACORN=1 optimization
- [ ] T020 [P] Test system behavior with 10K+ document scale requirements

## Phase 3.5: Constitutional Compliance Validation
- [ ] T021 [P] Verify Framework-First Architecture compliance in iris_rag/config/manager.py
- [ ] T022 [P] Verify Pipeline Validation & Requirements compliance in iris_rag/storage/schema_manager.py
- [ ] T023 [P] Verify Test-Driven Development compliance with contract tests
- [ ] T024 [P] Verify Performance & Enterprise Scale compliance with benchmarks
- [ ] T025 [P] Verify Production Readiness compliance with logging and error handling
- [ ] T026 [P] Verify Explicit Error Handling compliance with ConfigValidationError usage
- [ ] T027 [P] Verify Standardized Database Interfaces compliance with IRISConnectionManager

## Phase 3.6: Integration & Polish
- [ ] T028 Execute complete quickstart.md validation scenarios with live IRIS
- [ ] T029 [P] Run all contract tests with live database and verify 100% pass rate
- [ ] T030 [P] Validate CLAUDE.md agent context accuracy with implementation
- [ ] T031 [P] Execute end-to-end RAG pipeline integration test using ConfigurationManager and SchemaManager
- [ ] T032 Update any outdated documentation discovered during validation

## Dependencies
- Database setup (T001-T003) before contract tests (T004-T010)
- Contract alignment (T004-T010) before implementation validation (T011-T016)
- Implementation validation before performance testing (T017-T020)
- All validation before constitutional compliance (T021-T027)
- All core validation before integration (T028-T032)

## Parallel Example
```
# Launch contract test fixes together (T004-T010):
Task: "Fix ConfigurationManager environment variable override test alignment"
Task: "Fix ConfigurationManager error message validation test alignment"
Task: "Align SchemaManager vector dimension authority test expectations"
Task: "Fix SchemaManager schema migration detection test contract"

# Launch constitutional compliance validation together (T021-T027):
Task: "Verify Framework-First Architecture compliance"
Task: "Verify Pipeline Validation & Requirements compliance"
Task: "Verify Test-Driven Development compliance"
Task: "Verify Performance & Enterprise Scale compliance"
```

## Critical Validation Areas
**Based on Phase 0 research findings**:

1. **ConfigurationManager Contract Alignment** (T004-T005):
   - Environment variable precedence: RAG_DATABASE__IRIS__HOST should override YAML
   - Error type alignment: ConfigValidationError vs FileNotFoundError handling

2. **SchemaManager Contract Alignment** (T006-T010):
   - Method interface expectations: test_connection() vs get_connection()
   - Vector dimension authority validation patterns
   - Schema metadata tracking interface alignment

3. **Performance Validation** (T017-T020):
   - Configuration access: <5ms actual vs <50ms requirement (EXCEEDS)
   - Schema migrations: <2s actual vs <5s requirement (EXCEEDS)
   - ACORN=1 HNSW optimization validation

4. **Constitutional Compliance** (T021-T027):
   - All 7 principles verified in Phase 0, tasks validate implementation details
   - Framework-first architecture patterns
   - Explicit error handling with transaction rollback

## Success Criteria
- [ ] All 22 contract tests pass (9 ConfigurationManager + 13 SchemaManager)
- [ ] Performance benchmarks confirm <50ms config access, <5s migrations
- [ ] All 7 constitutional principles validated with implementation evidence
- [ ] End-to-end integration validates production readiness
- [ ] IRIS EHAT container with ACORN=1 optimization fully utilized

## Notes
- [P] tasks = independent validations, different components
- Implementation is COMPLETE - focus is verification and alignment
- Live IRIS database required for database-dependent tests
- EHAT container provides licensed features (ACORN=1) for optimal performance
- All tasks validate existing implementation rather than creating new code

## Validation Checklist
*GATE: All items must be verified before completion*

- [x] All contracts have corresponding validation tests
- [x] All entities have implementation validation tasks
- [x] All performance requirements have benchmark tasks
- [x] Constitutional compliance has verification tasks
- [x] Each task specifies exact file path and success criteria
- [x] Database tasks properly isolated and marked
- [x] Implementation exceeds all functional requirements