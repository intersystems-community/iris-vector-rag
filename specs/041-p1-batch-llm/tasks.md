# Tasks: Batch LLM Entity Extraction Integration

**Input**: Design documents from `/specs/041-p1-batch-llm/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/batch_extraction_api.md, quickstart.md

## Execution Flow
```
1. Load plan.md from feature directory
   → ✓ COMPLETE: Tech stack extracted (Python 3.11, DSPy, tiktoken, pytest)
2. Load design documents:
   → ✓ data-model.md: 3 entities (DocumentBatch, BatchExtractionResult, ProcessingMetrics)
   → ✓ contracts/: 4 API contracts (extract_batch, estimate_tokens, add_document/get_next_batch, get_statistics)
   → ✓ research.md: tiktoken library, collections.deque, custom retry logic
   → ✓ quickstart.md: 5 integration scenarios
3. Generate tasks by category:
   → Setup: dependencies (tiktoken), configuration
   → Tests: 3 contract test files + 5 integration test files + 3 unit test files
   → Core: 3 models + 4 utilities + service integration
   → Integration: DSPy module update, configuration
   → Polish: performance tests, documentation, optional clean IRIS validation
4. Apply task rules:
   → Different files = [P] parallel execution
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001-T026)
6. Return: SUCCESS (26 tasks ready for execution, 1 optional)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 3.1: Setup & Dependencies

- [x] T001 Add tiktoken dependency using `uv add tiktoken>=0.5.0` (per Constitution v1.6.0 package management requirement)
- [x] T002 [P] Create common/batch_utils.py stub file
- [x] T003 [P] Create iris_rag/utils/token_counter.py stub file
- [x] T004 Update config/memory_config.yaml to add batch_processing configuration section

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (all parallel - different files)
- [x] T005 [P] Contract test for extract_batch() in tests/contract/test_batch_extraction_contract.py
  - Validate signature (documents, token_budget params)
  - Validate return type (BatchExtractionResult)
  - Validate error on empty documents list

- [x] T006 [P] Contract test for token counter in tests/contract/test_token_counter_contract.py
  - Validate estimate_tokens() accuracy (±10% tolerance)
  - Validate empty string returns 0 tokens
  - Validate large document estimation (5000 words)

- [x] T007 [P] Contract test for batch queue in tests/contract/test_batch_queue_contract.py
  - Validate get_next_batch() respects token budget
  - Validate empty queue returns None
  - Validate add_document() queuing behavior

- [x] T008 [P] Contract test for batch metrics in tests/contract/test_batch_metrics_contract.py
  - Validate get_statistics() returns ProcessingMetrics
  - Validate FR-007 required fields (total_batches, avg_time, entity_rate, zero_entity_count)
  - Validate metrics update incrementally

### Integration Tests (all parallel - different files)
- [x] T009 [P] Integration test for 1K document batch processing in tests/integration/test_batch_extraction_e2e.py
  - Test AS-1: 1,000 documents → validate 3x speedup
  - Test AS-3: Entity traceability (document ID preservation)
  - Test AS-5: Single document → batch queue integration

- [x] T010 [P] Integration test for retry logic in tests/integration/test_batch_retry_logic.py
  - Test AS-2: Batch failure → exponential backoff (2s, 4s, 8s)
  - Test batch splitting after 3 failed retries
  - Test success after retry attempt

- [x] T011 [P] Integration test for variable document sizes in tests/integration/test_batch_sizing.py
  - Test AS-4: Variable sizes → dynamic batch sizing
  - Test token budget enforcement (8,192 default)
  - Test batch queue optimal packing

- [x] T012 [P] Integration test for performance validation in tests/integration/test_batch_performance.py
  - Test 1K documents speedup (target: 3.0x)
  - Test 10K documents speedup (target: 3.0x)
  - Test quality maintenance (4.86 entities/doc average)
  - Test mixed document types in same batch (FR-009 validation, if pipeline processes multiple types)

### Unit Tests (all parallel - different files)
- [x] T013 [P] Unit test for token counter in tests/unit/test_token_counter.py
  - Test tiktoken integration
  - Test different model encodings
  - Test edge cases (None, empty, special characters)

- [x] T014 [P] Unit test for batch queue in tests/unit/test_batch_queue.py
  - Test FIFO ordering
  - Test token budget calculations
  - Test queue state transitions

- [x] T015 [P] Unit test for batch sizing logic in tests/unit/test_batch_sizing.py
  - Test dynamic batch size calculation
  - Test token count accumulation
  - Test batch boundary conditions

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models (parallel - same file but independent models)
- [ ] T016 Add DocumentBatch model to iris_rag/core/models.py
  - Fields: batch_id, document_ids, batch_size, total_token_count, creation_timestamp, processing_status, retry_count
  - Methods: add_document(), is_within_budget()
  - Enum: BatchStatus (PENDING, PROCESSING, COMPLETED, FAILED, SPLIT)

- [ ] T017 Add BatchExtractionResult model to iris_rag/core/models.py (after T016)
  - Fields: batch_id, per_document_entities, per_document_relationships, processing_time, success_status, retry_count, error_message
  - Methods: get_all_entities(), get_all_relationships(), get_entity_count_by_document()

- [ ] T018 Add ProcessingMetrics model to iris_rag/core/models.py (after T017)
  - Fields: total_batches_processed, total_documents_processed, average_batch_processing_time, speedup_factor, entity_extraction_rate_per_batch, zero_entity_documents_count, failed_batches_count, retry_attempts_total
  - Methods: update_with_batch(), calculate_speedup()

### Utility Implementations
- [ ] T019 Implement token counter in iris_rag/utils/token_counter.py
  - Function: estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int
  - Use tiktoken.encoding_for_model()
  - Handle edge cases (empty, None, unsupported models)

- [ ] T020 Implement BatchQueue in common/batch_utils.py
  - Class: BatchQueue with collections.deque
  - Methods: add_document(document, token_count), get_next_batch(token_budget=8192)
  - Token-aware batching logic (from research.md)

- [ ] T021 Implement retry logic in common/batch_utils.py (after T020)
  - Function: extract_batch_with_retry(documents, extract_fn) -> BatchExtractionResult
  - Exponential backoff: 2s, 4s, 8s delays
  - Batch splitting on final failure

- [ ] T022 Implement BatchMetricsTracker in common/batch_utils.py (after T021)
  - Class: BatchMetricsTracker
  - Method: get_statistics() -> ProcessingMetrics
  - Global singleton pattern for metrics tracking

### Service Integration
- [ ] T023 Add extract_batch() method to iris_rag/services/entity_extraction.py
  - Method signature: extract_batch(documents: List[Document], token_budget: int = 8192) -> BatchExtractionResult
  - Integrate BatchQueue for document batching
  - Call DSPy BatchEntityExtractionModule
  - Use JSON retry logic (existing _parse_json_with_retry)
  - Wrap with retry logic from common/batch_utils.py

- [ ] T024 Add get_batch_metrics() method to iris_rag/services/entity_extraction.py (after T023)
  - Method signature: get_batch_metrics() -> ProcessingMetrics
  - Return global BatchMetricsTracker statistics

## Phase 3.4: Integration & Configuration

- [ ] T025 Update iris_rag/dspy_modules/batch_entity_extraction.py
  - Integrate JSON retry logic from entity_extraction.py
  - Replace json.loads() with _parse_json_with_retry()
  - Add batch-level logging (token count, processing time, retry count)

## Phase 3.5: Optional Validation (Constitution Best Practice)

- [ ] T026 [Optional] Clean IRIS validation test in tests/integration/test_batch_clean_iris_setup.py
  - Mark with @pytest.mark.clean_iris decorator
  - Start from empty IRIS database (no existing schema/data)
  - Run complete batch processing setup orchestration
  - Validate schema creation, configuration, and first batch execution
  - **Rationale**: Constitution principle III requires clean IRIS tests to validate self-sufficiency in new environments
  - **Note**: May be skipped if batch processing is considered an extension of existing entity extraction (not a new pipeline)

## Dependencies

**Critical TDD Dependencies**:
- Tests (T005-T015) MUST fail before ANY implementation (T016-T025)
- Contract tests validate API contracts exist
- Integration tests validate end-to-end scenarios
- Unit tests validate component behavior

**Implementation Dependencies**:
- T001 (tiktoken) blocks T019 (token counter implementation)
- T016-T018 (models) block T023 (service integration uses models)
- T019 (token counter) blocks T020 (BatchQueue uses token counting)
- T020 (BatchQueue) blocks T021 (retry logic uses BatchQueue)
- T022 (metrics tracker) blocks T024 (get_batch_metrics uses tracker)
- T023 (extract_batch) blocks T025 (DSPy integration uses extract_batch context)

**Test-Implementation Dependencies**:
- T005 validates T023 (extract_batch contract)
- T006 validates T019 (token counter contract)
- T007 validates T020 (batch queue contract)
- T008 validates T022, T024 (metrics contract)
- T009 validates T023 + T020 (e2e batch processing)
- T010 validates T021 (retry logic)
- T011 validates T020 (batch sizing)
- T012 validates T023 (performance targets)

## Parallel Execution Examples

### Phase 3.2 Contract Tests (Run together):
```
Task: "Contract test for extract_batch() in tests/contract/test_batch_extraction_contract.py"
Task: "Contract test for token counter in tests/contract/test_token_counter_contract.py"
Task: "Contract test for batch queue in tests/contract/test_batch_queue_contract.py"
Task: "Contract test for batch metrics in tests/contract/test_batch_metrics_contract.py"
```

### Phase 3.2 Integration Tests (Run together):
```
Task: "Integration test for 1K document batch processing in tests/integration/test_batch_extraction_e2e.py"
Task: "Integration test for retry logic in tests/integration/test_batch_retry_logic.py"
Task: "Integration test for variable document sizes in tests/integration/test_batch_sizing.py"
Task: "Integration test for performance validation in tests/integration/test_batch_performance.py"
```

### Phase 3.2 Unit Tests (Run together):
```
Task: "Unit test for token counter in tests/unit/test_token_counter.py"
Task: "Unit test for batch queue in tests/unit/test_batch_queue.py"
Task: "Unit test for batch sizing logic in tests/unit/test_batch_sizing.py"
```

### Phase 3.1 Setup (Run together):
```
Task: "Create common/batch_utils.py stub file"
Task: "Create iris_rag/utils/token_counter.py stub file"
```

## Notes

- **[P] tasks**: Different files, no dependencies - can run in parallel
- **TDD critical**: Verify all tests fail before implementing (T005-T015 before T016-T025)
- **Performance validation**: T012 must pass with 3.0x speedup to meet FR-002
- **Quality validation**: T012 must maintain 4.86 entities/doc average (FR-003)
- **Constitution compliance**: All tests require IRIS database per Constitution principle III (Test-Driven Development with Live Database Validation)
- **Commit strategy**: Commit after each task completion
- **Avoid**: Same file conflicts in parallel tasks, vague task descriptions

## Validation Checklist
*GATE: Verify before starting implementation*

- [x] All contracts have corresponding tests (T005-T008 cover 4 contracts)
- [x] All entities have model tasks (T016-T018 cover 3 models)
- [x] All tests come before implementation (T005-T015 before T016-T025)
- [x] Parallel tasks truly independent (verified file paths unique for [P] tasks)
- [x] Each task specifies exact file path (all tasks include absolute paths)
- [x] No task modifies same file as another [P] task (verified no conflicts)
- [x] Performance tests validate 3x speedup requirement (T012)
- [x] Quality tests validate 4.86 entities/doc (T012)
- [x] Integration tests cover all 5 user stories (AS-1 through AS-5)

## Success Criteria

**From spec.md Functional Requirements**:
- FR-001: Process 5-10 documents per batch → T023 (extract_batch implementation)
- FR-002: 3x speedup (7.7h → 2.5h) → T012 (performance validation)
- FR-003: Maintain 4.86 entities/doc quality → T012 (quality validation)
- FR-004: Entity traceability preserved → T009 (traceability test)
- FR-005: Exponential backoff retry → T010, T021 (retry logic)
- FR-006: Dynamic batch sizing (8K token budget) → T011, T020 (batch queue)
- FR-007: Processing statistics exposed → T008, T022, T024 (metrics API)

**Ready for implementation** ✅
