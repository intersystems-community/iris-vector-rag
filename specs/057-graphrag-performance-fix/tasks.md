# Tasks: GraphRAG Storage Performance Optimization

**Feature Branch**: `057-graphrag-performance-fix`
**Input**: Design documents from `/Users/tdyar/ws/iris-vector-rag-private/specs/057-graphrag-performance-fix/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/ (all complete)

## Overview

This feature optimizes GraphRAG storage performance from 60 seconds per ticket to 10-15 seconds, achieving 240-360 tickets/hour throughput (5-8x improvement). The optimization focuses on three critical bottlenecks: redundant embedding model loads (12-30 sec savings), serial entity storage without batching (30-64 sec savings), and IRIS connection overhead (3-7 sec savings).

**Key Performance Targets**:
- Individual ticket processing: ≤15 seconds (currently 60s)
- Throughput: ≥240 tickets/hour (currently 42/hour)
- Storage operations: ≤10 seconds (currently 50-120s)
- Complete dataset (10,150 tickets): ≤17 hours (currently 96 hours)

**Data Integrity Requirements**:
- 100% entity preservation (zero data loss)
- 100% relationship integrity (all foreign keys valid)
- Exact content preservation (SHA256 validation)

---

## Phase 1: Setup & Validation

- [ ] T001 Verify Python 3.11+ environment and iris-vector-rag dependencies installed
- [ ] T002 Verify IRIS database is running and accessible (connection test)
- [ ] T003 Verify DSPy entity extraction is configured and working (5-6 second baseline)
- [ ] T004 Verify SentenceTransformer model `all-MiniLM-L6-v2` is downloaded and cached

---

## Phase 2: Foundational - Contract Tests (TDD) ⚠️ MUST COMPLETE BEFORE PHASE 3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Performance Contracts (4 tests, PC-001 to PC-004)
- [ ] T005 [P] Contract test PC-001: Single ticket processing ≤15 seconds in `tests/contract/test_performance_contract.py::test_pc001_single_ticket_15_seconds`
- [ ] T006 [P] Contract test PC-002: Throughput ≥240 tickets/hour in `tests/contract/test_performance_contract.py::test_pc002_throughput_240_per_hour`
- [ ] T007 [P] Contract test PC-003: Storage operations ≤10 seconds in `tests/contract/test_performance_contract.py::test_pc003_storage_10_seconds`
- [ ] T008 [P] Contract test PC-004: Dataset completion ≤17 hours in `tests/contract/test_performance_contract.py::test_pc004_dataset_17_hours`

### Data Integrity Contracts (3 tests, DIC-001 to DIC-003)
- [ ] T009 [P] Contract test DIC-001: 100% entity preservation in `tests/contract/test_data_integrity_contract.py::test_dic001_no_entity_loss`
- [ ] T010 [P] Contract test DIC-002: Entity content match (SHA256) in `tests/contract/test_data_integrity_contract.py::test_dic002_entity_content_match`
- [ ] T011 [P] Contract test DIC-003: Relationship integrity in `tests/contract/test_data_integrity_contract.py::test_dic003_relationship_integrity`

### Monitoring Contracts (4 tests, MC-001 to MC-004)
- [ ] T012 [P] Contract test MC-001: Millisecond precision tracking in `tests/contract/test_monitoring_contract.py::test_mc001_millisecond_precision`
- [ ] T013 [P] Contract test MC-002: Real-time throughput tracking in `tests/contract/test_monitoring_contract.py::test_mc002_realtime_throughput`
- [ ] T014 [P] Contract test MC-003: Alert on slow ticket (>20s) in `tests/contract/test_monitoring_contract.py::test_mc003_slow_ticket_alert`
- [ ] T015 [P] Contract test MC-004: Timing breakdown logging in `tests/contract/test_monitoring_contract.py::test_mc004_timing_breakdowns`

**Validation Gate**: Run all contract tests with `pytest tests/contract/` - ALL tests MUST FAIL before proceeding to Phase 3

---

## Phase 3: Core Optimization - UnifiedEmbeddingService (Priority 1: 12-30 sec savings)

**Goal**: Eliminate redundant embedding model loads by consolidating to single cached instance

- [ ] T016 Verify `iris_rag/embeddings/manager.py` has `_SENTENCE_TRANSFORMER_CACHE` with thread-safe double-check locking
- [ ] T017 [P] Integrate UnifiedEmbeddingService into `iris_rag/services/storage.py` (entity storage adapter)
- [ ] T018 [P] Integrate UnifiedEmbeddingService into `iris_rag/embeddings/entity_extractor.py` (pattern extraction service)
- [ ] T019 [P] Integrate UnifiedEmbeddingService into `iris_rag/services/entity_extraction.py` (relationship processing)
- [ ] T020 [P] Integrate UnifiedEmbeddingService into `iris_rag/embeddings/iris_embedding.py` (memory creation service)
- [ ] T021 Verify unified embedding service eliminates model reloads (check logs for "Using cached model")

**Expected Impact**: 12-30 seconds saved per ticket (eliminate 4-6 model loads × 3-5 seconds each)

---

## Phase 4: Core Optimization - BatchEntityProcessor (Priority 2: 30-64 sec savings)

**Goal**: Replace serial entity storage with batch processing using `executemany()` and single transaction per ticket

- [ ] T022 Create `iris_rag/services/batch_entity_processor.py` with batch storage logic
  - Implement `store_entities_batch(entities, batch_size=32)` method
  - Use IRIS DBAPI `cursor.executemany()` for batch inserts
  - Single transaction boundary per ticket (commit after all entities stored)
  - Include entity count validation (extracted count == stored count)
- [ ] T023 Update `iris_rag/services/storage.py` to use BatchEntityProcessor instead of serial storage
- [ ] T024 Implement relationship batch storage in `batch_entity_processor.py`
  - `store_relationships_batch(relationships, batch_size=32)` method
  - Foreign key validation (all entity IDs exist)
  - Zero orphaned relationships check
- [ ] T025 Add error handling and rollback logic for batch failures
  - Automatic retry with smaller batch size on connection timeout
  - Transaction rollback on validation failure
  - Clear error messages with context

**Expected Impact**: 30-64 seconds saved per ticket (batch vs serial: 6-10s total instead of 40-70s)

---

## Phase 5: Core Optimization - ConnectionPooling (Priority 3: 3-7 sec savings)

**Goal**: Prevent connection overhead by implementing connection pool with pre-ping validation

- [ ] T026 Create `iris_rag/common/iris_connection_manager.py` with connection pooling
  - Implement `IRISConnectionPool` with 20 base connections + 10 overflow
  - Connection recycling with 1-hour age limit
  - Pre-ping validation before each use (health check)
  - Thread-safe acquire/release with timeout handling
- [ ] T027 Update `iris_rag/services/storage.py` to use connection pool instead of new connection per request
- [ ] T028 Update `iris_rag/services/batch_entity_processor.py` to use connection pool
- [ ] T029 Update `iris_rag/embeddings/iris_embedding.py` to use connection pool
- [ ] T030 Add connection pool monitoring and health checks
  - Track active/idle connections
  - Log pool exhaustion events
  - Alert when wait time exceeds threshold

**Expected Impact**: 3-7 seconds saved per ticket (eliminate 1-2s connection overhead × multiple operations)

---

## Phase 6: Monitoring & Validation

**Goal**: Implement performance monitoring and data integrity validation

### Performance Monitoring
- [ ] T031 Create `iris_rag/common/performance_monitor.py` with non-blocking metric collection
  - Background monitoring thread (daemon=True, 5-second interval)
  - In-memory deque with 1000-entry circular buffer (no DB writes during monitoring)
  - Real-time throughput calculation (tickets/hour from rolling window)
  - Alert thresholds: >20s per ticket triggers warning
- [ ] T032 Implement `record_query_performance(response_time_ms, cache_hit, db_time, hnsw_time)` method
  - Thread-safe metric recording with lock
  - Millisecond precision timestamps (ISO 8601 with 3 decimal places)
  - Structured JSON logging with timing breakdowns
- [ ] T033 Integrate performance monitoring into `iris_rag/services/storage.py`
  - Record extraction_time_ms, storage_time_ms, total_time_ms
  - Log at INFO level with ticket_id correlation
  - Calculate and log tickets_per_hour in real-time

### Data Integrity Validation
- [ ] T034 Create `iris_rag/validation/data_integrity_validator.py` with post-storage validation
  - Entity count validation: `extracted_count == stored_count`
  - Relationship foreign key validation (SQL query for orphaned relationships)
  - Sample-based content hash verification (10% spot checks with SHA256)
- [ ] T035 Integrate data integrity validator into `iris_rag/services/batch_entity_processor.py`
  - Validate after batch storage (non-blocking, <30ms overhead)
  - Raise DataIntegrityError on mismatch with actionable context
  - Log validation results at INFO level

---

## Phase 7: Integration Tests

**Goal**: Validate end-to-end processing with all optimizations integrated

- [ ] T036 [P] Integration test: Unified embedding service eliminates model reloads in `tests/integration/test_unified_embedding_integration.py`
- [ ] T037 [P] Integration test: Batch entity storage processes 10 entities in ≤10s in `tests/integration/test_batch_entity_storage.py`
- [ ] T038 [P] Integration test: Connection pooling prevents overhead in `tests/integration/test_connection_pooling.py`
- [ ] T039 [P] Integration test: Performance monitoring records metrics correctly in `tests/integration/test_performance_monitoring.py`
- [ ] T040 [P] Integration test: Data integrity validation catches errors in `tests/integration/test_data_integrity_validation.py`

---

## Phase 8: Performance Testing & Validation

**Goal**: Validate sustained performance under realistic load

### Performance Benchmarks
- [ ] T041 Create 100-ticket throughput test in `tests/performance/test_100_ticket_throughput.py`
  - Process 100 continuous tickets with 8-12 entities each
  - Measure: Total time, throughput (tickets/hour), average time per ticket
  - Success criteria: ≤25 minutes total, ≥240 tickets/hour, avg ≤15s per ticket
- [ ] T042 Create 1000-ticket sustained load test in `tests/performance/test_1000_ticket_sustained.py`
  - Process 1000 tickets continuously
  - Monitor: Memory usage, CPU usage, throughput stability
  - Success criteria: Throughput ≥240/hour sustained, no memory leaks, stable performance
- [ ] T043 Run 100-ticket throughput test and verify contract PC-002 passes
- [ ] T044 Run 1000-ticket sustained test and verify extrapolated dataset completion ≤17 hours (PC-004)

### Contract Validation
- [ ] T045 Run all performance contract tests (PC-001 to PC-004) - ALL MUST PASS
- [ ] T046 Run all data integrity contract tests (DIC-001 to DIC-003) - ALL MUST PASS
- [ ] T047 Run all monitoring contract tests (MC-001 to MC-004) - ALL MUST PASS

**Validation Gate**: All 11 contract tests MUST PASS before proceeding to Phase 9

---

## Phase 9: Polish & Documentation

**Goal**: Finalize documentation and validate deployment readiness

- [ ] T048 [P] Update `iris_rag/README.md` with performance optimization details
- [ ] T049 [P] Update `CLAUDE.md` with new batch processing patterns and performance monitoring usage
- [ ] T050 [P] Create performance comparison table (baseline vs optimized) in `specs/057-graphrag-performance-fix/RESULTS.md`
- [ ] T051 Validate quickstart workflow in `specs/057-graphrag-performance-fix/quickstart.md`
  - Run Steps 1-6 validation workflow
  - Verify all success criteria checkboxes can be checked
- [ ] T052 Create production deployment checklist
  - Performance validated (all PC contracts passing)
  - Data integrity confirmed (all DIC contracts passing)
  - Monitoring active (all MC contracts passing)
  - Rollback plan documented
  - Operator approval obtained

---

## Dependencies

### Sequential Dependencies (must complete in order)
- Phase 1 (Setup) → Phase 2 (Contract Tests)
- Phase 2 → Phase 3, 4, 5 (Core Optimizations)
- T022 (batch_entity_processor.py) blocks T023, T024, T025, T028, T035
- T026 (connection pooling) blocks T027, T028, T029, T030
- T031 (performance_monitor.py) blocks T032, T033
- T034 (data_integrity_validator.py) blocks T035
- Phase 3, 4, 5 → Phase 6 (Monitoring)
- Phase 6 → Phase 7 (Integration Tests)
- Phase 7 → Phase 8 (Performance Testing)
- T041, T042 (performance tests created) → T043, T044 (run tests)
- T043, T044 → T045, T046, T047 (contract validation)
- Phase 8 → Phase 9 (Polish)

### Parallel Opportunities (independent tasks, different files)
- **Phase 2 (Contract Tests)**: T005-T015 can all run in parallel (11 tasks, different test files)
- **Phase 3 (UnifiedEmbedding)**: T017-T020 can run in parallel (4 tasks, different service files)
- **Phase 7 (Integration Tests)**: T036-T040 can run in parallel (5 tasks, different test files)
- **Phase 9 (Documentation)**: T048-T050 can run in parallel (3 tasks, different doc files)

---

## Parallel Execution Examples

### Phase 2: All Contract Tests Together (11 parallel tasks)
```bash
# Launch T005-T015 in parallel (different test files, no dependencies)
pytest tests/contract/test_performance_contract.py::test_pc001_single_ticket_15_seconds &
pytest tests/contract/test_performance_contract.py::test_pc002_throughput_240_per_hour &
pytest tests/contract/test_performance_contract.py::test_pc003_storage_10_seconds &
pytest tests/contract/test_performance_contract.py::test_pc004_dataset_17_hours &
pytest tests/contract/test_data_integrity_contract.py::test_dic001_no_entity_loss &
pytest tests/contract/test_data_integrity_contract.py::test_dic002_entity_content_match &
pytest tests/contract/test_data_integrity_contract.py::test_dic003_relationship_integrity &
pytest tests/contract/test_monitoring_contract.py::test_mc001_millisecond_precision &
pytest tests/contract/test_monitoring_contract.py::test_mc002_realtime_throughput &
pytest tests/contract/test_monitoring_contract.py::test_mc003_slow_ticket_alert &
pytest tests/contract/test_monitoring_contract.py::test_mc004_timing_breakdowns &
wait
```

### Phase 3: Unified Embedding Integration (4 parallel tasks)
```bash
# Launch T017-T020 in parallel (different service files, no dependencies)
# T017: Update iris_rag/services/storage.py
# T018: Update iris_rag/embeddings/entity_extractor.py
# T019: Update iris_rag/services/entity_extraction.py
# T020: Update iris_rag/embeddings/iris_embedding.py
```

### Phase 7: Integration Tests (5 parallel tasks)
```bash
# Launch T036-T040 in parallel (different test files, no dependencies)
pytest tests/integration/test_unified_embedding_integration.py &
pytest tests/integration/test_batch_entity_storage.py &
pytest tests/integration/test_connection_pooling.py &
pytest tests/integration/test_performance_monitoring.py &
pytest tests/integration/test_data_integrity_validation.py &
wait
```

---

## MVP Scope Recommendation

**MVP (Minimum Viable Product)**: Phase 1-5 + Phase 8 Contract Validation
- Phase 1: Setup & validation (4 tasks)
- Phase 2: Contract tests (11 tasks) - TDD requirement
- Phase 3: UnifiedEmbeddingService integration (6 tasks) - 12-30 sec savings
- Phase 4: BatchEntityProcessor implementation (4 tasks) - 30-64 sec savings
- Phase 5: ConnectionPooling implementation (5 tasks) - 3-7 sec savings
- Phase 8 (partial): Contract validation (T045-T047) - 3 tasks

**Total MVP Tasks**: 33 tasks
**Expected MVP Impact**: 45-101 seconds saved per ticket (75-87% improvement)
**MVP Completion Criteria**: All 11 contract tests passing (PC-001 to MC-004)

**Post-MVP (Nice-to-Have)**:
- Phase 6: Monitoring & Validation (5 tasks) - Enhances observability
- Phase 7: Integration Tests (5 tasks) - Additional test coverage
- Phase 8 (remaining): Performance benchmarks (4 tasks) - Sustained load validation
- Phase 9: Polish & Documentation (5 tasks) - Production readiness

---

## Task Summary

| Phase | Task Count | Parallel Tasks | MVP Priority |
|-------|------------|----------------|--------------|
| Phase 1: Setup | 4 | 0 | P0 (Required) |
| Phase 2: Contract Tests | 11 | 11 | P0 (TDD Gate) |
| Phase 3: UnifiedEmbedding | 6 | 4 | P0 (12-30s savings) |
| Phase 4: BatchProcessor | 4 | 0 | P0 (30-64s savings) |
| Phase 5: ConnectionPool | 5 | 0 | P0 (3-7s savings) |
| Phase 6: Monitoring | 5 | 0 | P1 (Observability) |
| Phase 7: Integration Tests | 5 | 5 | P1 (Test Coverage) |
| Phase 8: Performance Testing | 7 | 0 | P0 (Validation) |
| Phase 9: Polish | 5 | 3 | P2 (Documentation) |
| **TOTAL** | **52** | **23** | **33 MVP tasks** |

---

## Validation Checklist

*GATE: Checked before feature is considered complete*

- [ ] All 4 performance contracts (PC-001 to PC-004) PASSING
- [ ] All 3 data integrity contracts (DIC-001 to DIC-003) PASSING
- [ ] All 4 monitoring contracts (MC-001 to MC-004) PASSING
- [ ] 100-ticket throughput test shows ≥240 tickets/hour
- [ ] 1000-ticket sustained test shows stable performance (no memory leaks)
- [ ] Performance comparison table shows 75-87% improvement
- [ ] Quickstart validation workflow completes successfully
- [ ] Production deployment checklist complete

---

## Notes

**TDD Principle**: Phase 2 contract tests MUST be written first and MUST FAIL before any implementation (Phase 3-5) begins. This ensures tests are validating actual behavior, not implementation details.

**Parallel Execution**: Tasks marked [P] can run in parallel because they modify different files and have no dependencies. Use this to accelerate development (23 of 52 tasks are parallelizable).

**Performance Targets**: This feature achieves 5-8x throughput improvement (42 → 240-360 tickets/hour) by addressing three bottlenecks identified in research.md:
1. **UnifiedEmbeddingService** (Priority 1): 12-30 sec savings
2. **BatchEntityProcessor** (Priority 2): 30-64 sec savings
3. **ConnectionPooling** (Priority 3): 3-7 sec savings

**Data Integrity**: Zero data loss requirement enforced via 3 contract tests (DIC-001 to DIC-003) with post-storage validation. Validation adds <30ms overhead (0.5% of 10-second target).

**Monitoring**: Real-time performance tracking via non-blocking background thread. Metrics collected in-memory (deque with 1000-entry buffer) to avoid database write overhead during critical path.

---

**Generated**: 2025-11-12
**Feature Branch**: 057-graphrag-performance-fix
**Constitution Version**: 1.8.0
