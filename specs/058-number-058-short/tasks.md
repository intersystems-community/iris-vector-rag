# Tasks: Cloud Configuration Flexibility

**Feature**: 058-cloud-config-flexibility
**Branch**: `058-number-058-short`
**Input**: Design documents from `/specs/058-number-058-short/`

## Overview

This feature enables flexible cloud deployment configuration for iris-vector-rag by implementing environment variable support, respecting config file specifications, and supporting variable vector dimensions (128-8192). Addresses 9 documented pain points from FHIR-AI-Hackathon-Kit AWS migration.

**Expected Impact**: Reduce cloud deployment time from 65 minutes to under 25 minutes (60% reduction).

## Task Format

```
- [ ] [TaskID] [P?] Description with file path
```

- **[P]**: Parallelizable (different files, no dependencies)
- **TaskID**: Sequential number (T001, T002...)
- Include exact file paths for all implementation tasks

## Phase 1: Setup & Validation

**Goal**: Verify existing codebase structure and prepare for configuration enhancements

- [ ] T001 Verify existing ConfigurationManager in iris_vector_rag/config/manager.py
- [ ] T002 Review existing environment variable pattern from Feature 035 (backend_modes.py)
- [ ] T003 Confirm pydantic is in dependencies (pyproject.toml) for validation
- [ ] T004 [P] Create test config file examples directory: config/examples/

## Phase 2: Contract Tests (TDD)

**CRITICAL: All contract tests MUST be written and MUST FAIL before ANY implementation**

**Goal**: Define expected behavior through failing tests (22 contract tests total)

### Environment Variable Configuration (FR-001)
- [ ] T005 [P] Contract test: IRIS_HOST environment variable in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_iris_host_from_environment_variable
- [ ] T006 [P] Contract test: IRIS_PORT environment variable in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_iris_port_from_environment_variable
- [ ] T007 [P] Contract test: IRIS_NAMESPACE environment variable in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_iris_namespace_from_environment_variable
- [ ] T008 [P] Contract test: All IRIS connection vars supported in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_all_iris_connection_vars_supported

### Configuration Priority (FR-005)
- [ ] T009 [P] Contract test: Env var overrides config file in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_environment_variable_overrides_config_file
- [ ] T010 [P] Contract test: Config file overrides defaults in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_config_file_overrides_defaults
- [ ] T011 [P] Contract test: Defaults used when no overrides in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_defaults_used_when_no_overrides

### Vector Dimension Configuration (FR-003)
- [ ] T012 [P] Contract test: Vector dimension via env var in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_vector_dimension_configurable_via_env_var
- [ ] T013 [P] Contract test: Vector dimension min validation in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_vector_dimension_validation_min_bound
- [ ] T014 [P] Contract test: Vector dimension max validation in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_vector_dimension_validation_max_bound
- [ ] T015 [P] Contract test: Common dimensions supported in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_common_vector_dimensions_supported

### Table Schema Configuration (FR-004)
- [ ] T016 [P] Contract test: Table schema via env var in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_table_schema_configurable_via_env_var
- [ ] T017 [P] Contract test: %SYS namespace support in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_table_schema_supports_percent_sys_namespace
- [ ] T018 [P] Contract test: Full table name with schema prefix in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_full_table_name_includes_schema_prefix

### init_tables Config Respect (FR-002, SC-007)
- [ ] T019 [P] Contract test: init_tables loads --config flag in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_init_tables_loads_config_from_flag
- [ ] T020 [P] Contract test: init_tables uses defaults without flag in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_init_tables_without_config_flag_uses_defaults

### Preflight Validation (FR-006)
- [ ] T021 [P] Contract test: Vector dimension mismatch detected in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_vector_dimension_mismatch_detected
- [ ] T022 [P] Contract test: Namespace permission validation in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_namespace_permission_validation

### Backward Compatibility (FR-008, SC-006)
- [ ] T023 [P] Contract test: v0.4.x defaults match in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_default_configuration_matches_v04x_behavior
- [ ] T024 [P] Contract test: Existing code works unchanged in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_existing_code_works_without_modifications

### Configuration Logging (FR-011)
- [ ] T025 [P] Contract test: Configuration source tracking in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_configuration_source_tracking
- [ ] T026 [P] Contract test: Password masking in logs in specs/058-number-058-short/contracts/test_cloud_config_contract.py::test_password_masking_in_configuration_log

**Checkpoint**: Run all contract tests - ALL MUST FAIL at this point

```bash
pytest specs/058-number-058-short/contracts/ -v
# Expected: 22 skipped tests (marked with @pytest.mark.skip)
```

## Phase 3: Core Configuration Entities

**Goal**: Implement pydantic models for configuration management

- [ ] T027 [P] Create ConnectionConfiguration model in iris_vector_rag/config/entities.py
- [ ] T028 [P] Create VectorConfiguration model in iris_vector_rag/config/entities.py
- [ ] T029 [P] Create TableConfiguration model with computed properties in iris_vector_rag/config/entities.py
- [ ] T030 [P] Create ConfigurationSource model for audit tracking in iris_vector_rag/config/entities.py

**Checkpoint**: Verify pydantic models with basic unit tests

## Phase 4: Configuration Validators

**Goal**: Implement preflight validation logic

- [ ] T031 [P] Create VectorDimensionValidator class in iris_vector_rag/config/validators.py
- [ ] T032 Implement validate() method for VectorDimensionValidator in iris_vector_rag/config/validators.py
- [ ] T033 [P] Create NamespaceValidator class in iris_vector_rag/config/validators.py
- [ ] T034 Implement validate() method for NamespaceValidator in iris_vector_rag/config/validators.py
- [ ] T035 [P] Create ValidationResult dataclass in iris_vector_rag/config/validators.py

**Checkpoint**: Validators can be instantiated and have correct interfaces

## Phase 5: ConfigurationManager Enhancements

**Goal**: Extend existing ConfigurationManager with new capabilities (FR-001, FR-005, FR-006)

- [ ] T036 Add IRIS connection environment variables support to ConfigurationManager._load_env_variables() in iris_vector_rag/config/manager.py
- [ ] T037 Implement configuration priority resolution (env > file > defaults) in iris_vector_rag/config/manager.py
- [ ] T038 Add VECTOR_DIMENSION and TABLE_SCHEMA environment variable support in iris_vector_rag/config/manager.py
- [ ] T039 Integrate VectorDimensionValidator into ConfigurationManager initialization in iris_vector_rag/config/manager.py
- [ ] T040 Integrate NamespaceValidator into ConfigurationManager initialization in iris_vector_rag/config/manager.py
- [ ] T041 Implement get_configuration_sources() method for audit logging in iris_vector_rag/config/manager.py
- [ ] T042 Add password masking logic to ConfigurationSource tracking in iris_vector_rag/config/manager.py

**Checkpoint**: Run contract tests T005-T011, T025-T026 - should start passing

```bash
pytest specs/058-number-058-short/contracts/ -k "environment_variable or priority or configuration_source" -v
```

## Phase 6: ConnectionManager Integration

**Goal**: Use enhanced configuration in connection management

- [ ] T043 Update ConnectionManager to use ConnectionConfiguration entity in iris_vector_rag/core/connection.py
- [ ] T044 Read IRIS_HOST, IRIS_PORT from ConfigurationManager in iris_vector_rag/core/connection.py
- [ ] T045 Read IRIS_USERNAME, IRIS_PASSWORD, IRIS_NAMESPACE from ConfigurationManager in iris_vector_rag/core/connection.py
- [ ] T046 Update connection string builder to use namespace from config in iris_vector_rag/core/connection.py

**Checkpoint**: Connection with environment variables works

## Phase 7: Vector Dimension Flexibility

**Goal**: Support configurable vector dimensions (FR-003)

- [ ] T047 Update embedding_config.py to read VECTOR_DIMENSION from ConfigurationManager in iris_vector_rag/config/embedding_config.py
- [ ] T048 Add vector dimension validation (128-8192 range) in iris_vector_rag/config/embedding_config.py
- [ ] T049 Update EntityStorageAdapter to use configured vector dimension in iris_vector_rag/services/storage.py
- [ ] T050 Update table creation SQL to use dynamic vector dimension in iris_vector_rag/services/storage.py

**Checkpoint**: Run contract tests T012-T015 - should pass

```bash
pytest specs/058-number-058-short/contracts/ -k "vector_dimension" -v
```

## Phase 8: Table Schema Configuration

**Goal**: Support schema-prefixed table names (FR-004)

- [ ] T051 Add TABLE_SCHEMA configuration support in iris_vector_rag/config/manager.py
- [ ] T052 Update EntityStorageAdapter to use TableConfiguration.full_entities_table in iris_vector_rag/services/storage.py
- [ ] T053 Update EntityStorageAdapter to use TableConfiguration.full_relationships_table in iris_vector_rag/services/storage.py
- [ ] T054 Update all SQL queries to use schema-prefixed table names in iris_vector_rag/services/storage.py
- [ ] T055 Ensure CREATE TABLE statements use configured schema in iris_vector_rag/services/storage.py

**Checkpoint**: Run contract tests T016-T018 - should pass

```bash
pytest specs/058-number-058-short/contracts/ -k "table_schema" -v
```

## Phase 9: init_tables CLI Fix

**Goal**: Fix init_tables to respect --config flag (FR-002, SC-007)

- [ ] T056 Add --config argument parser in iris_vector_rag/cli/init_tables.py
- [ ] T057 Pass config_path to ConfigurationManager instantiation in iris_vector_rag/cli/init_tables.py
- [ ] T058 Use configured vector_dimension from ConfigurationManager in iris_vector_rag/cli/init_tables.py
- [ ] T059 Use configured table_schema from ConfigurationManager in iris_vector_rag/cli/init_tables.py
- [ ] T060 Display configuration summary before table creation in iris_vector_rag/cli/init_tables.py

**Checkpoint**: Run contract tests T019-T020 - should pass

```bash
pytest specs/058-number-058-short/contracts/ -k "init_tables" -v
# Also manual test:
python -m iris_vector_rag.cli.init_tables --help  # Should show --config flag
```

## Phase 10: Preflight Validation Integration

**Goal**: Add validation before operations (FR-006)

- [ ] T061 Add preflight validation call in ConnectionManager.get_connection() in iris_vector_rag/core/connection.py
- [ ] T062 Add preflight validation call in init_tables before table creation in iris_vector_rag/cli/init_tables.py
- [ ] T063 Implement clear error messages with migration guidance in iris_vector_rag/config/validators.py
- [ ] T064 Add help URLs to validation error messages in iris_vector_rag/config/validators.py

**Checkpoint**: Run contract tests T021-T022 - should pass

```bash
pytest specs/058-number-058-short/contracts/ -k "validation" -v
```

## Phase 11: Configuration Examples & Documentation

**Goal**: Provide cloud-specific configuration templates

- [ ] T065 [P] Create AWS IRIS configuration template in config/examples/aws.yaml
- [ ] T066 [P] Create Azure IRIS configuration template in config/examples/azure.yaml
- [ ] T067 [P] Create local development configuration template in config/examples/local.yaml
- [ ] T068 [P] Update default_config.yaml with cloud deployment examples in iris_vector_rag/config/default_config.yaml
- [ ] T069 [P] Add inline comments explaining each configuration option in config/examples/*.yaml

**Checkpoint**: Configuration examples are complete and valid YAML

## Phase 12: Integration Testing

**Goal**: Validate complete workflow against live IRIS database

- [ ] T070 [P] Integration test: Environment variable configuration flow in tests/integration/test_config_integration.py
- [ ] T071 [P] Integration test: Config file priority override in tests/integration/test_config_integration.py
- [ ] T072 [P] Integration test: Vector dimension validation against real IRIS in tests/integration/test_config_integration.py
- [ ] T073 [P] Integration test: Namespace permission check against real IRIS in tests/integration/test_config_integration.py
- [ ] T074 [P] Integration test: Schema-prefixed table creation in real IRIS in tests/integration/test_config_integration.py
- [ ] T075 [P] Integration test: init_tables --config flag end-to-end in tests/integration/test_config_integration.py

**Checkpoint**: Run integration tests with live IRIS connection

```bash
# Requires IRIS running on localhost:1972 or environment variables set
pytest tests/integration/test_config_integration.py -v
```

## Phase 13: Backward Compatibility Verification

**Goal**: Ensure 100% backward compatibility (FR-008, SC-006)

- [ ] T076 Verify default configuration matches v0.4.x behavior in tests/integration/test_config_integration.py
- [ ] T077 Run existing test suite to confirm no regressions: `pytest tests/unit/ tests/integration/ -v`
- [ ] T078 Test existing pipelines work with no environment variables or config files
- [ ] T079 Verify ConnectionManager works with existing code patterns

**Checkpoint**: All existing tests pass, zero breaking changes

## Phase 14: Final Contract Test Validation

**Goal**: Verify all 22 contract tests pass

- [ ] T080 Remove @pytest.mark.skip from all contract tests in specs/058-number-058-short/contracts/test_cloud_config_contract.py
- [ ] T081 Run complete contract test suite: `pytest specs/058-number-058-short/contracts/ -v`
- [ ] T082 Verify 22/22 contract tests passing (0 skipped, 0 failed)

**Expected**: All 22 contract tests passing

## Phase 15: Polish & Documentation

**Goal**: Finalize feature with docs and cleanup

- [ ] T083 [P] Update CLAUDE.md with configuration examples
- [ ] T084 [P] Update README.md with quickstart guide for cloud deployment
- [ ] T085 [P] Create docs/cloud-deployment.md with comprehensive guide
- [ ] T086 [P] Create docs/troubleshooting/configuration-errors.md
- [ ] T087 [P] Add configuration validation performance benchmark in tests/performance/
- [ ] T088 Review and remove any code duplication in config/ module
- [ ] T089 Run linting: `black . && isort . && flake8 iris_vector_rag/config/`
- [ ] T090 Final verification: Run quickstart.md test scenarios manually

**Checkpoint**: Feature is production-ready

## Dependencies

**Critical Path**:
1. Setup (T001-T004)
2. Contract Tests (T005-T026) - MUST complete before implementation
3. Core Entities (T027-T030)
4. Validators (T031-T035)
5. ConfigurationManager (T036-T042)
6. Integration (T043-T064)
7. Documentation (T065-T069)
8. Testing (T070-T082)
9. Polish (T083-T090)

**Parallel Opportunities**:
- All contract tests (T005-T026) can run in parallel
- All entity models (T027-T030) can be built in parallel
- Both validators (T031, T033) can be built in parallel
- Configuration examples (T065-T069) can be created in parallel
- Integration tests (T070-T075) can run in parallel (different test methods)
- Documentation tasks (T083-T086) can be done in parallel

**Blocking Dependencies**:
- ConfigurationManager enhancements (T036-T042) block all integration (T043+)
- Contract tests MUST fail before implementation begins
- Integration testing (T070-T075) requires Phases 3-10 complete
- Final validation (T080-T082) requires all implementation complete

## Parallel Execution Examples

### Contract Tests (Phase 2)
```bash
# All 22 contract tests in parallel (different test methods):
pytest specs/058-number-058-short/contracts/test_cloud_config_contract.py -v -n auto
```

### Core Entities (Phase 3)
```bash
# All 4 entity models in parallel (same file, different classes):
# Task: "Create ConnectionConfiguration model in iris_vector_rag/config/entities.py"
# Task: "Create VectorConfiguration model in iris_vector_rag/config/entities.py"
# Task: "Create TableConfiguration model in iris_vector_rag/config/entities.py"
# Task: "Create ConfigurationSource model in iris_vector_rag/config/entities.py"
```

### Configuration Examples (Phase 11)
```bash
# All example files in parallel (different files):
# Task: "Create AWS IRIS configuration template in config/examples/aws.yaml"
# Task: "Create Azure IRIS configuration template in config/examples/azure.yaml"
# Task: "Create local development template in config/examples/local.yaml"
```

## Success Criteria Validation

**From Specification**:
- **SC-001**: Cloud deployment time reduces from 65 minutes to under 25 minutes (60% reduction)
  - Validation: Manual deployment test with AWS IRIS using environment variables
- **SC-002**: Zero code modifications required to deploy to AWS IRIS, Azure, or GCP
  - Validation: Test T078 verifies existing code works + quickstart examples require no code changes
- **SC-003**: Users can switch embedding models (384-dim to 1024-dim) through configuration only
  - Validation: Contract tests T012-T015 + integration test T072
- **SC-006**: Existing local deployments continue working without any changes (100% backward compatible)
  - Validation: Phase 13 (T076-T079) verifies backward compatibility
- **SC-007**: init_tables() respects --config flag 100% of the time (currently 0%)
  - Validation: Contract tests T019-T020 + integration test T075

## Implementation Strategy

### MVP Scope (Recommended First PR)
**Minimal viable implementation** - Phases 1-10 only:
- Setup (T001-T004)
- Contract Tests (T005-T026)
- Core implementation (T027-T064)
- Skip integration tests, examples, and polish

**Rationale**: Get core functionality working and validated via contract tests. Can iterate on examples and docs in follow-up PRs.

### Incremental Delivery
1. **PR 1**: Phases 1-10 (Core configuration flexibility) - ~35 tasks
2. **PR 2**: Phase 11 (Configuration examples) - ~5 tasks
3. **PR 3**: Phases 12-13 (Integration testing + backward compat) - ~10 tasks
4. **PR 4**: Phases 14-15 (Final validation + polish) - ~11 tasks

### Testing Strategy
- **TDD Approach**: All 22 contract tests written first (Phase 2)
- **Green Light**: Contract tests pass incrementally as features are implemented
- **Integration Validation**: Phase 12 tests against real IRIS database
- **Regression Prevention**: Phase 13 ensures zero breaking changes

## Notes

- **TDD Critical**: Contract tests MUST be written and failing before implementation starts
- **Parallel Markers**: [P] tasks can be executed concurrently (different files, no dependencies)
- **Commit Frequency**: Commit after completing each phase (not individual tasks)
- **Configuration Priority**: Environment variables > config file > defaults (12-factor app pattern)
- **Backward Compatibility**: ALL changes are opt-in with sensible defaults
- **Performance Target**: Configuration validation overhead < 100ms at startup

---

**Total Tasks**: 90
**Parallel Tasks**: 35 (marked with [P])
**Estimated Effort**: 35-45 hours for complete implementation
**MVP Effort**: 20-25 hours (Phases 1-10 only)

**Ready for Execution**: All tasks are specific, have clear acceptance criteria, and include exact file paths.
