# Tasks: Enterprise Enhancements for RAG System

**Input**: Design documents from `/specs/051-enterprise-enhancements/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included per constitutional requirement (TDD approach - line 51 of plan.md)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- Python package structure: `iris_vector_rag/` at repository root
- Tests: `tests/contract/`, `tests/integration/`, `tests/unit/`
- Config: `config/`, `iris_vector_rag/config/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency installation

- [ ] T001 Add OpenTelemetry dependencies to pyproject.toml as optional extras (monitoring, enterprise)
- [ ] T002 [P] Install dependencies via `uv sync --extra enterprise` or `pip install -e .[enterprise]`
- [ ] T003 [P] Create iris_vector_rag/security/ module with __init__.py
- [ ] T004 [P] Create iris_vector_rag/monitoring/ module with __init__.py
- [ ] T005 [P] Create iris_vector_rag/exceptions.py for new exception types

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Create VectorStoreConfigurationError exception class in iris_vector_rag/exceptions.py
- [ ] T007 [P] Create PermissionDeniedError exception class in iris_vector_rag/exceptions.py
- [ ] T008 [P] Extend ConfigurationManager in iris_vector_rag/config/manager.py to handle new config sections (storage.iris.custom_filter_keys, security.rbac, telemetry, batch_operations)
- [ ] T009 Update iris_vector_rag/config/default_config.yaml with new configuration sections per research.md decision 1
- [ ] T010 [P] Create constants.py in iris_vector_rag/storage/ for DEFAULT_FILTER_KEYS list (17 fields)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Custom Metadata Filtering for Multi-Tenancy (Priority: P1) 🎯 MVP

**Goal**: Enable filtering documents by custom metadata fields (tenant_id, security_level, department) for multi-tenant deployments

**Independent Test**: Configure custom metadata fields via YAML, query with custom filters, verify correct document filtering and clear error messages for unconfigured fields

### Tests for User Story 1 (TDD - Write FIRST, ensure they FAIL)

- [ ] T011 [P] [US1] Contract test: test_custom_field_configuration in tests/contract/test_metadata_filtering_contract.py
- [ ] T012 [P] [US1] Contract test: test_metadata_filter_validation_success in tests/contract/test_metadata_filtering_contract.py
- [ ] T013 [P] [US1] Contract test: test_metadata_filter_validation_failure in tests/contract/test_metadata_filtering_contract.py
- [ ] T014 [P] [US1] Contract test: test_duplicate_field_name_rejection in tests/contract/test_metadata_filtering_contract.py
- [ ] T015 [P] [US1] Contract test: test_invalid_field_name_rejection in tests/contract/test_metadata_filtering_contract.py
- [ ] T016 [P] [US1] Contract test: test_empty_custom_fields_backward_compatibility in tests/contract/test_metadata_filtering_contract.py
- [ ] T017 [P] [US1] Contract test: test_case_sensitive_field_names in tests/contract/test_metadata_filtering_contract.py
- [ ] T018 [P] [US1] Contract test: test_special_characters_in_field_values (SQL injection prevention) in tests/contract/test_metadata_filtering_contract.py
- [ ] T019 [P] [US1] Integration test: test_custom_metadata_filters_e2e in tests/integration/test_custom_metadata_filters.py
- [ ] T020 [P] [US1] Unit test: test_metadata_filter_manager in tests/unit/storage/test_metadata_filter_manager.py

### Implementation for User Story 1

- [ ] T021 [P] [US1] Create MetadataFilterManager class in iris_vector_rag/storage/metadata_filter_manager.py with validate_field_name() method per research.md decision 1
- [ ] T022 [US1] Implement MetadataFilterManager.get_allowed_filter_keys() returning {default_keys, custom_keys, all_keys}
- [ ] T023 [US1] Implement MetadataFilterManager.validate_filter_keys() to check against allowed list
- [ ] T024 [US1] Extend IRISVectorStore.__init__() in iris_vector_rag/storage/vector_store_iris.py to initialize MetadataFilterManager
- [ ] T025 [US1] Add IRISVectorStore.get_allowed_filter_keys() method exposing allowed keys
- [ ] T026 [US1] Modify IRISVectorStore.similarity_search() to validate metadata_filter keys before SQL execution
- [ ] T027 [US1] Add clear error handling: raise VectorStoreConfigurationError with rejected_keys and allowed_keys on invalid filter
- [ ] T028 [US1] Update iris_vector_rag/core/models.py to document metadata_filter parameter in query signatures
- [ ] T029 [US1] Run contract tests for US1 - verify all pass

**Checkpoint**: At this point, User Story 1 should be fully functional - custom metadata filtering works, SQL injection prevented, backward compatible

---

## Phase 4: User Story 2 - Collection Lifecycle Management (Priority: P1)

**Goal**: Provide CRUD operations for document collections (list, get info, create, delete, check existence)

**Independent Test**: Create multiple collections, list them with statistics, get details for specific collection, delete collection, check existence

### Tests for User Story 2 (TDD - Write FIRST, ensure they FAIL)

- [ ] T030 [P] [US2] Contract test: test_list_collections_success in tests/contract/test_collection_management_contract.py
- [ ] T031 [P] [US2] Contract test: test_list_collections_empty in tests/contract/test_collection_management_contract.py
- [ ] T032 [P] [US2] Contract test: test_list_collections_performance (<2s for 1000 collections) in tests/contract/test_collection_management_contract.py
- [ ] T033 [P] [US2] Contract test: test_get_collection_info_success in tests/contract/test_collection_management_contract.py
- [ ] T034 [P] [US2] Contract test: test_get_collection_info_not_found in tests/contract/test_collection_management_contract.py
- [ ] T035 [P] [US2] Contract test: test_create_collection_success in tests/contract/test_collection_management_contract.py
- [ ] T036 [P] [US2] Contract test: test_create_collection_duplicate in tests/contract/test_collection_management_contract.py
- [ ] T037 [P] [US2] Contract test: test_create_collection_invalid_name in tests/contract/test_collection_management_contract.py
- [ ] T038 [P] [US2] Contract test: test_delete_collection_success in tests/contract/test_collection_management_contract.py
- [ ] T039 [P] [US2] Contract test: test_delete_collection_not_found in tests/contract/test_collection_management_contract.py
- [ ] T040 [P] [US2] Contract test: test_collection_exists_true in tests/contract/test_collection_management_contract.py
- [ ] T041 [P] [US2] Contract test: test_collection_exists_false in tests/contract/test_collection_management_contract.py
- [ ] T042 [P] [US2] Contract test: test_collection_lifecycle_integration in tests/contract/test_collection_management_contract.py
- [ ] T043 [P] [US2] Contract test: test_collection_metadata_update in tests/contract/test_collection_management_contract.py
- [ ] T044 [P] [US2] Integration test: test_collection_lifecycle in tests/integration/test_collection_lifecycle.py

### Implementation for User Story 2

- [ ] T045 [P] [US2] Add IRISVectorStore.list_collections() method in iris_vector_rag/storage/vector_store_iris.py returning List[Dict] with statistics
- [ ] T046 [P] [US2] Add IRISVectorStore.get_collection_info(collection_id) method returning Collection entity dict
- [ ] T047 [P] [US2] Add IRISVectorStore.create_collection(collection_id, metadata) method for explicit collection creation
- [ ] T048 [P] [US2] Add IRISVectorStore.delete_collection(collection_id) method returning deleted document count
- [ ] T049 [P] [US2] Add IRISVectorStore.collection_exists(collection_id) method returning boolean
- [ ] T050 [US2] Implement SQL queries for collection statistics (document_count, total_size_bytes, created_at, last_updated)
- [ ] T051 [US2] Add validation for collection_id format (alphanumeric + hyphens/underscores, max 128 chars)
- [ ] T052 [US2] Add error handling for non-existent collections (clear error messages)
- [ ] T053 [US2] Run contract tests for US2 - verify all pass

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently - collection management fully functional

---

## Phase 5: User Story 3 - Permission-Based Access Control (Priority: P1)

**Goal**: Integrate RBAC policy interface for collection-level and document-level access control

**Independent Test**: Implement mock RBAC policy, configure it, verify users with different permissions only access authorized collections and documents

### Tests for User Story 3 (TDD - Write FIRST, ensure they FAIL)

- [ ] T054 [P] [US3] Contract test: test_rbac_disabled_backward_compatibility in tests/contract/test_rbac_integration_contract.py
- [ ] T055 [P] [US3] Contract test: test_collection_read_permission_granted in tests/contract/test_rbac_integration_contract.py
- [ ] T056 [P] [US3] Contract test: test_collection_read_permission_denied in tests/contract/test_rbac_integration_contract.py
- [ ] T057 [P] [US3] Contract test: test_document_level_filtering in tests/contract/test_rbac_integration_contract.py
- [ ] T058 [P] [US3] Contract test: test_write_permission_granted in tests/contract/test_rbac_integration_contract.py
- [ ] T059 [P] [US3] Contract test: test_write_permission_denied in tests/contract/test_rbac_integration_contract.py
- [ ] T060 [P] [US3] Contract test: test_delete_permission_granted in tests/contract/test_rbac_integration_contract.py
- [ ] T061 [P] [US3] Contract test: test_delete_permission_denied in tests/contract/test_rbac_integration_contract.py
- [ ] T062 [P] [US3] Contract test: test_audit_context_enrichment in tests/contract/test_rbac_integration_contract.py
- [ ] T063 [P] [US3] Contract test: test_permission_check_error_handling in tests/contract/test_rbac_integration_contract.py
- [ ] T064 [P] [US3] Contract test: test_clear_error_messages in tests/contract/test_rbac_integration_contract.py
- [ ] T065 [P] [US3] Integration test: test_rbac_enforcement_e2e in tests/integration/test_rbac_enforcement.py
- [ ] T066 [P] [US3] Unit test: test_rbac_policy_interface in tests/unit/security/test_rbac_policy.py

### Implementation for User Story 3

- [ ] T067 [P] [US3] Create RBACPolicy abstract base class in iris_vector_rag/security/rbac.py per research.md decision 2
- [ ] T068 [US3] Implement RBACPolicy.check_collection_access(user, collection_id, operation) abstract method
- [ ] T069 [US3] Implement RBACPolicy.filter_documents(user, documents) abstract method
- [ ] T070 [US3] Implement RBACPolicy.get_audit_context(user) optional method with default implementation
- [ ] T071 [US3] Create PermissionDeniedError exception with user, resource, operation fields in iris_vector_rag/exceptions.py
- [ ] T072 [US3] Extend IRISVectorStore.__init__() to accept optional rbac_policy parameter
- [ ] T073 [US3] Add RBAC permission check to IRISVectorStore.similarity_search() before query execution
- [ ] T074 [US3] Add RBAC permission check to IRISVectorStore.add_documents() before insertion
- [ ] T075 [US3] Add RBAC permission check to IRISVectorStore.delete_collection() before deletion
- [ ] T076 [US3] Implement document-level filtering after retrieval using policy.filter_documents()
- [ ] T077 [US3] Add exception handling for RBAC policy errors: wrap check_collection_access() in try/except, raise PermissionDeniedError("Permission check failed: {error}") on exceptions
- [ ] T078 [US3] Add audit logging integration (call get_audit_context() when policy present)
- [ ] T079 [US3] Ensure backward compatibility: RBAC checks skipped when policy=None
- [ ] T080 [US3] Create example MockRBACPolicy for testing in tests/fixtures/mock_rbac_policy.py
- [ ] T081 [US3] Run contract tests for US3 - verify all pass

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently - RBAC integration functional, P1 features complete

---

## Phase 6: User Story 4 - Production Observability and Monitoring (Priority: P2)

**Goal**: Integrate OpenTelemetry instrumentation for query latency, token usage, and cost tracking with zero overhead when disabled

**Independent Test**: Enable monitoring, execute queries and document indexing, verify telemetry data collected with correct spans and attributes, measure overhead (<5% enabled, 0% disabled)

### Tests for User Story 4 (TDD - Write FIRST, ensure they FAIL)

- [ ] T092 [P] [US4] Contract test: test_telemetry_disabled_zero_overhead in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_telemetry_enabled_low_overhead (<5%) in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_query_operation_span_creation in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_retrieval_operation_span in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_generation_operation_span in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_embedding_generation_span in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_cost_tracking in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_error_recording in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_trace_context_propagation in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_sampling_ratio in tests/contract/test_monitoring_contract.py
- [ ] T092 [P] [US4] Contract test: test_otlp_export in tests/contract/test_monitoring_contract.py
- [ ] T093 [P] [US4] Integration test: test_telemetry_spans_e2e in tests/integration/test_telemetry_spans.py
- [ ] T094 [P] [US4] Unit test: test_telemetry_manager in tests/unit/monitoring/test_telemetry.py

### Implementation for User Story 4

- [ ] T095 [P] [US4] Create TelemetryManager class in iris_vector_rag/monitoring/telemetry.py per research.md decision 3
- [ ] T096 [US4] Implement TelemetryManager.__init__(enabled, config) with lazy initialization: if enabled=False, store config only (no SDK initialization); if enabled=True, initialize TracerProvider, OTLP exporter, and sampler on first trace_operation() call
- [ ] T097 [US4] Implement TelemetryManager.trace_operation() context manager with early return when disabled
- [ ] T098 [US4] Implement TelemetryManager.get_status() returning current config and stats
- [ ] T099 [US4] Create configure_telemetry(enabled, service_name, endpoint) function
- [ ] T100 [US4] Create global telemetry singleton instance (initialized disabled by default)
- [ ] T101 [US4] Instrument IRISVectorStore.similarity_search() with telemetry.trace_operation("rag.retrieval")
- [ ] T102 [US4] Instrument embedding generation with telemetry.trace_operation("rag.embedding")
- [ ] T103 [US4] Instrument LLM generation with telemetry.trace_operation("rag.generation") and GenAI attributes
- [ ] T104 [US4] Create calculate_llm_cost() function in iris_vector_rag/monitoring/cost_tracking.py with pricing table
- [ ] T105 [US4] Add span attributes following OpenTelemetry GenAI semantic conventions (gen.ai.*)
- [ ] T106 [US4] Implement error recording in spans (span.set_status, span.record_exception)
- [ ] T107 [US4] Add OTLP exporter configuration (HTTP endpoint, sampling ratio)
- [ ] T108 [US4] Run performance benchmarks: verify 0% overhead when disabled, <5% when enabled
- [ ] T109 [US4] Run contract tests for US4 - verify all pass

**Checkpoint**: At this point, User Stories 1-4 should work independently - monitoring operational, P2 feature 1 complete

---

## Phase 7: User Story 5 - Bulk Document Loading (Priority: P2)

**Goal**: Implement bulk document loading with 10x+ performance improvement, progress tracking, and configurable error handling

**Independent Test**: Load 10,000 documents via bulk operation, measure time (<10s), compare to one-by-one loading, verify error handling strategies (continue/stop/rollback)

### Tests for User Story 5 (TDD - Write FIRST, ensure they FAIL)

- [ ] T110 [P] [US5] Contract test: test_bulk_loading_success (10K docs <10s) in tests/contract/test_batch_operations_contract.py
- [ ] T111 [P] [US5] Contract test: test_bulk_loading_with_errors_continue in tests/contract/test_batch_operations_contract.py
- [ ] T112 [P] [US5] Contract test: test_bulk_loading_stop_on_error in tests/contract/test_batch_operations_contract.py
- [ ] T113 [P] [US5] Contract test: test_bulk_loading_rollback_on_error in tests/contract/test_batch_operations_contract.py
- [ ] T114 [P] [US5] Contract test: test_progress_percentage_updates in tests/contract/test_batch_operations_contract.py
- [ ] T115 [P] [US5] Contract test: test_batch_size_configuration in tests/contract/test_batch_operations_contract.py
- [ ] T116 [P] [US5] Contract test: test_throughput_calculation in tests/contract/test_batch_operations_contract.py
- [ ] T117 [P] [US5] Contract test: test_error_limit_max_100 in tests/contract/test_batch_operations_contract.py
- [ ] T118 [P] [US5] Contract test: test_pre_computed_embeddings in tests/contract/test_batch_operations_contract.py
- [ ] T119 [P] [US5] Contract test: test_empty_document_list in tests/contract/test_batch_operations_contract.py
- [ ] T120 [P] [US5] Contract test: test_invalid_batch_size in tests/contract/test_batch_operations_contract.py
- [ ] T121 [P] [US5] Contract test: test_performance_vs_one_by_one (10x+ speedup) in tests/contract/test_batch_operations_contract.py
- [ ] T122 [P] [US5] Integration test: test_bulk_loading_e2e in tests/integration/test_bulk_loading.py
- [ ] T123 [P] [US5] Unit test: test_batch_operations in tests/unit/storage/test_batch_operations.py

### Implementation for User Story 5

- [ ] T124 [P] [US5] Create BatchOperations class in iris_vector_rag/storage/batch_operations.py per research.md decision 4
- [ ] T125 [US5] Implement IRISVectorStore.add_documents_batch(documents, embeddings, batch_size, show_progress, error_handling)
- [ ] T126 [US5] Implement streaming strategy: process documents in batches of 1000 (default)
- [ ] T127 [US5] Implement "continue" error handling: skip failed documents, log errors, continue processing (all batches including partial in-progress batch)
- [ ] T128 [US5] Implement "stop" error handling: stop on first error, commit completed full batches, rollback in-progress partial batch
- [ ] T129 [US5] Implement "rollback" error handling: stop on first error, rollback ALL batches using savepoints (all-or-nothing semantics)
- [ ] T130 [US5] Add progress tracking: calculate progress_percentage after each batch
- [ ] T131 [US5] Add tqdm progress bar support when show_progress=True
- [ ] T132 [US5] Implement _insert_batch() helper for parameterized batch INSERT
- [ ] T133 [US5] Implement _insert_batch_with_individual_errors() for continue mode
- [ ] T134 [US5] Add support for pre-computed embeddings (skip generation if provided)
- [ ] T135 [US5] Calculate and return throughput_docs_per_sec in result
- [ ] T136 [US5] Limit errors array to max 100 entries in response
- [ ] T137 [US5] Add validation: batch_size >= 1, documents non-empty
- [ ] T138 [US5] Run performance benchmark: 10K docs must complete in <10s with batch_size=1000
- [ ] T139 [US5] Run contract tests for US5 - verify all pass

**Checkpoint**: At this point, User Stories 1-5 should work independently - bulk loading operational, P2 features complete

---

## Phase 8: User Story 6 - Metadata Schema Discovery (Priority: P3)

**Goal**: Discover metadata schema from document collections via statistical sampling (100-200 docs) with type inference and statistics

**Independent Test**: Sample documents from collection, verify correct schema inference with types, frequencies, examples, and statistics

### Tests for User Story 6 (TDD - Write FIRST, ensure they FAIL)

- [ ] T140 [P] [US6] Contract test: test_schema_discovery_string_field in tests/contract/test_schema_discovery_contract.py
- [ ] T141 [P] [US6] Contract test: test_schema_discovery_integer_field in tests/contract/test_schema_discovery_contract.py
- [ ] T142 [P] [US6] Contract test: test_schema_discovery_datetime_field in tests/contract/test_schema_discovery_contract.py
- [ ] T143 [P] [US6] Contract test: test_schema_discovery_performance (<5s) in tests/contract/test_schema_discovery_contract.py
- [ ] T144 [P] [US6] Contract test: test_schema_discovery_sample_size_100 in tests/contract/test_schema_discovery_contract.py
- [ ] T145 [P] [US6] Contract test: test_schema_discovery_all_collections in tests/contract/test_schema_discovery_contract.py
- [ ] T146 [P] [US6] Integration test: test_schema_discovery_e2e in tests/integration/test_schema_discovery.py
- [ ] T147 [P] [US6] Unit test: test_type_inference in tests/unit/storage/test_schema_discovery.py

### Implementation for User Story 6

- [ ] T148 [P] [US6] Implement IRISVectorStore.sample_metadata_schema(collection_id, sample_size) in iris_vector_rag/storage/vector_store_iris.py per research.md decision 5
- [ ] T149 [US6] Implement stratified random sampling using IRIS SQL (ORDER BY NEWID())
- [ ] T150 [US6] Implement _infer_type(value) helper: detect string/integer/float/datetime/boolean/array/object
- [ ] T151 [US6] Aggregate metadata fields: count occurrences, track types, collect values
- [ ] T152 [US6] Calculate frequency (occurrence rate) for each field
- [ ] T153 [US6] Calculate statistics for numeric fields: min, max, avg
- [ ] T154 [US6] Collect example values for string/datetime fields (max 5)
- [ ] T155 [US6] Determine primary type (most common) for each field
- [ ] T156 [US6] Build and return schema dictionary with all field info
- [ ] T157 [US6] Add datetime detection logic (ISO 8601 format strings)
- [ ] T158 [US6] Run performance test: schema discovery with sample_size=200 must complete in <5s
- [ ] T159 [US6] Run contract tests for US6 - verify all pass

**Checkpoint**: All user stories should now be independently functional - metadata schema discovery complete, P3 feature done

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T160 [P] Update pyproject.toml with correct version number and new dependencies
- [ ] T161 [P] Update CHANGELOG.md with v0.6.0 enterprise enhancements release notes
- [ ] T162 [P] Add docstrings to all new classes and public methods
- [ ] T163 [P] Update README.md with enterprise features section and installation instructions
- [ ] T164 [P] Create example RBAC policy implementations in examples/security/ (LDAPRBACPolicy, ClearanceLevelPolicy)
- [ ] T165 [P] Create example telemetry configuration in examples/monitoring/telemetry_config.yaml
- [ ] T166 [P] Add type hints to all new functions and methods
- [ ] T167 [P] Run black and isort on all modified files
- [ ] T168 [P] Run flake8 linting - fix any issues
- [ ] T169 [P] Run mypy type checking - fix any issues
- [ ] T170 Run full test suite: pytest tests/ --cov=iris_vector_rag
- [ ] T171 Validate quickstart.md examples work end-to-end
- [ ] T172 [P] Security audit: verify SQL injection prevention in metadata filters
- [ ] T173 [P] Performance validation: run all performance benchmarks (metadata filtering <5ms, collection list <2s, bulk load <10s, monitoring overhead <5%)
- [ ] T174 Update API documentation in iris_vector_rag/api/README.md with new endpoints
- [ ] T175 Create migration guide for existing users upgrading to v0.6.0

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-8)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order: US1 (P1) → US2 (P1) → US3 (P1) → US4 (P2) → US5 (P2) → US6 (P3)
- **Polish (Phase 9)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 5 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 6 (P3)**: Can start after Foundational (Phase 2) - No dependencies on other stories

**Key Insight**: All 6 user stories are independently testable and can be implemented in parallel once Foundational phase completes!

### Within Each User Story

- Tests MUST be written and FAIL before implementation (TDD approach)
- Contract tests → Implementation → Integration tests → Unit tests
- All tests for a story can run in parallel (marked [P])
- Implementation tasks follow logical sequence within story

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T002-T005)
- All Foundational tasks marked [P] can run in parallel (T007, T010)
- Once Foundational phase completes, **all 6 user stories can start in parallel** (if team capacity allows)
- All tests within a user story marked [P] can run in parallel
- Polish tasks marked [P] can run in parallel (T160-T169, T172-T173)

---

## Parallel Example: User Story 1

```bash
# Launch all contract tests for User Story 1 together:
Task T011: "Contract test: test_custom_field_configuration"
Task T012: "Contract test: test_metadata_filter_validation_success"
Task T013: "Contract test: test_metadata_filter_validation_failure"
Task T014: "Contract test: test_duplicate_field_name_rejection"
Task T015: "Contract test: test_invalid_field_name_rejection"
Task T016: "Contract test: test_empty_custom_fields_backward_compatibility"
Task T017: "Contract test: test_case_sensitive_field_names"
Task T018: "Contract test: test_special_characters_in_field_values"
Task T019: "Integration test: test_custom_metadata_filters_e2e"
Task T020: "Unit test: test_metadata_filter_manager"

# Launch implementation tasks for User Story 1:
Task T021: "Create MetadataFilterManager class" [P]
# Then after T021 completes, launch dependent tasks
```

---

## Parallel Example: Multiple User Stories

```bash
# After Foundational phase (T001-T010) completes:

# Team Member 1: User Story 1 (Custom Metadata Filtering)
Tasks T011-T029

# Team Member 2: User Story 2 (Collection Management)
Tasks T030-T053

# Team Member 3: User Story 3 (RBAC Integration)
Tasks T054-T080

# All three stories can proceed simultaneously without conflicts!
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T005)
2. Complete Phase 2: Foundational (T006-T010) - CRITICAL - blocks all stories
3. Complete Phase 3: User Story 1 (T011-T029)
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo custom metadata filtering feature

### Incremental Delivery (Recommended)

1. Complete Setup + Foundational → Foundation ready (T001-T010)
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!) (T011-T029)
3. Add User Story 2 → Test independently → Deploy/Demo (T030-T053)
4. Add User Story 3 → Test independently → Deploy/Demo (T054-T080)
5. **P1 Features Complete** - Critical enterprise features done
6. Add User Story 4 → Test independently → Deploy/Demo (T081-T109)
7. Add User Story 5 → Test independently → Deploy/Demo (T110-T139)
8. **P2 Features Complete** - High-value operational features done
9. Add User Story 6 → Test independently → Deploy/Demo (T140-T159)
10. **All Features Complete** - Polish and release (T160-T175)

Each story adds value without breaking previous stories!

### Parallel Team Strategy (6 Developers)

With 6 developers:

1. **Team completes Setup + Foundational together** (T001-T010)
2. **Once Foundational is done, stories diverge:**
   - Developer A: User Story 1 (Custom Metadata Filtering) - T011-T029
   - Developer B: User Story 2 (Collection Management) - T030-T053
   - Developer C: User Story 3 (RBAC Integration) - T054-T080
   - Developer D: User Story 4 (Monitoring) - T081-T109
   - Developer E: User Story 5 (Bulk Loading) - T110-T139
   - Developer F: User Story 6 (Schema Discovery) - T140-T159
3. **Stories complete and integrate independently**
4. **Team regroups for Polish phase** (T160-T175)

---

## Task Summary

- **Total Tasks**: 174
- **Setup Phase**: 5 tasks
- **Foundational Phase**: 5 tasks (BLOCKING)
- **User Story 1 (P1)**: 19 tasks (10 tests + 9 implementation)
- **User Story 2 (P1)**: 24 tasks (15 tests + 9 implementation)
- **User Story 3 (P1)**: 27 tasks (13 tests + 14 implementation)
- **User Story 4 (P2)**: 28 tasks (13 tests + 15 implementation)
- **User Story 5 (P2)**: 30 tasks (14 tests + 16 implementation)
- **User Story 6 (P3)**: 20 tasks (8 tests + 12 implementation)
- **Polish Phase**: 16 tasks

**Parallelizable Tasks**: 145 tasks marked [P] can run in parallel (83% of tasks!)

**Independent User Stories**: All 6 stories are independently testable and deployable

**Suggested MVP Scope**: User Story 1 only (19 tasks) - provides critical multi-tenant filtering capability

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability (US1-US6)
- Each user story should be independently completable and testable
- TDD approach: Verify tests fail before implementing (constitutional requirement)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All enhancements maintain 100% backward compatibility (features default to disabled)
- Performance targets must be validated: metadata filtering <5ms, collection list <2s, bulk load <10s, monitoring overhead <5%
- Security critical: SQL injection prevention in custom metadata filters (test T018)
