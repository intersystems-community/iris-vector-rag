# Implementation Plan: GraphRAG Storage Performance Optimization

**Branch**: `057-graphrag-performance-fix` | **Date**: 2025-11-12 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/057-graphrag-performance-fix/spec.md`

## Summary

The GraphRAG storage pipeline is experiencing 80-87% performance degradation, processing tickets at 60 seconds each instead of the expected 10-15 seconds. LLM entity extraction is fast (5-6 seconds), but IRIS storage operations consume 50-120 seconds per ticket due to redundant embedding model loads, serial entity storage without batching, and IRIS connection/transaction overhead. This feature optimizes the storage layer to achieve 240-360 tickets/hour throughput (5-8x improvement), reducing complete dataset processing time from 96 hours to 11-17 hours.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**:
- iris-vector-rag framework (existing)
- SentenceTransformer (`all-MiniLM-L6-v2`) for embeddings
- InterSystems IRIS Community Edition
- DSPy for entity extraction (no changes needed)

**Storage**: InterSystems IRIS database (vector + graph storage)
**Testing**: pytest with contract tests, integration tests, performance benchmarks
**Target Platform**: Linux server (dpgenai1.iscinternal.com production environment)
**Project Type**: Single project (iris-vector-rag framework enhancement)

**Performance Goals**:
- Individual ticket processing: ≤15 seconds (currently 60s)
- Throughput: ≥240 tickets/hour (currently 42/hour)
- IRIS storage operations: ≤10 seconds (currently 50-120s)
- Complete dataset (10,150 tickets): ≤17 hours (currently 96 hours)

**Constraints**:
- Zero data loss or corruption (100% data integrity required)
- No changes to entity extraction logic (already meeting performance targets)
- Backward compatibility with existing storage format
- Memory usage must remain stable (no leaks)
- Must support graceful shutdown and resume capabilities

**Scale/Scope**:
- Production dataset: 10,150 tickets
- Entity volume: 8-12 entities per ticket (82,000-122,000 total)
- Relationship volume: 4-6 relationships per ticket (41,000-61,000 total)
- Performance testing required for 100-ticket batches and 1,000-ticket sustained load

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component enhances existing iris-vector-rag framework services
- ✓ No application-specific logic (pure framework optimization)
- ✓ CLI interface exposed via existing Make targets and service commands

**II. Pipeline Validation & Requirements**:
- ✓ Automated requirement validation exists in framework
- ✓ Setup procedures remain idempotent (no changes to validation)

**III. Test-Driven Development**:
- ✓ Contract tests required before implementation
- ✓ Performance tests for 100-ticket and 1,000-ticket scenarios mandatory
- ✓ Data integrity validation tests required

**IV. Performance & Enterprise Scale**:
- ✓ This feature IMPLEMENTS batch entity storage optimization
- ✓ IRIS vector operations will be optimized via batching and connection pooling
- ✓ Addresses enterprise scale requirement (10K+ documents)

**V. Production Readiness**:
- ✓ Structured logging exists, will add throughput monitoring
- ✓ Health checks exist in framework
- ✓ Docker deployment already supported

**VI. Explicit Error Handling**:
- ✓ No silent failures (framework already enforces this)
- ✓ Clear exception messages for storage failures
- ✓ Actionable error context for performance degradation

**VII. Standardized Database Interfaces**:
- ✓ Will use existing iris_rag database utilities
- ✓ Batch operations will extend proven patterns
- ✓ Optimizations will be contributed back to shared utilities

**Initial Constitution Check: PASS** ✅

## Project Structure

### Documentation (this feature)
```
specs/057-graphrag-performance-fix/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   ├── performance_contracts.yaml
│   ├── data_integrity_contracts.yaml
│   └── monitoring_contracts.yaml
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_rag/
├── services/
│   ├── unified_embedding_service.py    # ✅ EXISTS (needs integration)
│   ├── entity_storage.py               # ⚠️ NEEDS BATCH OPTIMIZATION
│   └── batch_entity_processor.py       # 🆕 NEW (batch processing logic)
├── embeddings/
│   └── manager.py                       # ⚠️ NEEDS UNIFIED SERVICE INTEGRATION
├── memory/
│   └── pattern_extractor.py            # ⚠️ NEEDS UNIFIED SERVICE INTEGRATION
├── relationships/
│   └── relationship_processor.py       # ⚠️ NEEDS UNIFIED SERVICE INTEGRATION
├── common/
│   ├── iris_connection_manager.py      # ⚠️ NEEDS CONNECTION POOLING
│   └── performance_monitor.py          # 🆕 NEW (throughput tracking)
└── validation/
    └── data_integrity_validator.py     # 🆕 NEW (post-optimization validation)

tests/
├── contract/
│   ├── test_performance_contract.py    # 🆕 NEW (FR-001 to FR-004)
│   ├── test_data_integrity_contract.py # 🆕 NEW (FR-005 to FR-007)
│   └── test_monitoring_contract.py     # 🆕 NEW (FR-008 to FR-011)
├── integration/
│   ├── test_batch_entity_storage.py    # 🆕 NEW
│   ├── test_unified_embedding_integration.py # 🆕 NEW
│   └── test_connection_pooling.py      # 🆕 NEW
└── performance/
    ├── test_100_ticket_throughput.py   # 🆕 NEW
    └── test_1000_ticket_sustained.py   # 🆕 NEW
```

**Structure Decision**: Single project enhancement to iris-vector-rag framework. New components follow existing framework patterns in `iris_rag/services/` and `iris_rag/common/`. Tests organized by contract/integration/performance tiers per constitutional requirements.

## Phase 0: Outline & Research

**Research Tasks** (all NEEDS CLARIFICATION items from Technical Context have been resolved via performance issue report):

1. **Embedding Batch Processing Best Practices**
   - Research: SentenceTransformer batch encoding patterns
   - Research: Optimal batch sizes for `all-MiniLM-L6-v2` model
   - Research: Memory usage patterns for batch operations

2. **IRIS Batch Operations**
   - Research: IRIS SQL batching patterns (executemany, prepared statements)
   - Research: Transaction boundaries for batch writes
   - Research: Connection pooling with InterSystems IRIS

3. **Performance Monitoring Patterns**
   - Research: Real-time throughput monitoring (non-blocking)
   - Research: Performance metric collection without overhead
   - Research: Alert thresholds for performance degradation

4. **Data Integrity Validation**
   - Research: Entity count validation strategies
   - Research: Relationship integrity checking patterns
   - Research: Content hash verification methods

**Output**: research.md consolidating findings with decisions, rationales, and alternatives considered.

## Phase 1: Design & Contracts

*Prerequisites: research.md complete*

### 1. Extract Entities → data-model.md

**Key Entities** (from spec.md):

**Ticket**:
- Fields: ticket_id (string), content_text (string), processing_timestamp (datetime), status (enum: pending/processing/completed/failed)
- Relationships: HasMany Entity (8-12 per ticket)
- State transitions: pending → processing → completed/failed

**Entity**:
- Fields: entity_id (string), entity_text (string), entity_type (string), extraction_source_ticket_id (string), embedding_vector (float[])
- Relationships: BelongsTo Ticket, HasMany Relationship (4-6 relationships)
- Validation: entity_text must be non-empty, embedding_vector dimension matches model

**Relationship**:
- Fields: relationship_id (string), relationship_type (string), source_entity_id (string), target_entity_id (string), confidence_score (float 0-1)
- Relationships: References two Entity records
- Validation: confidence_score between 0.0 and 1.0, entity IDs must exist

**ProcessingMetrics**:
- Fields: metric_id (string), timestamp (datetime), ticket_id (string), extraction_time_ms (int), storage_time_ms (int), total_time_ms (int), success (boolean)
- Purpose: Real-time performance tracking
- Validation: All time fields ≥0, total_time = extraction_time + storage_time

### 2. Generate API Contracts → /contracts/

**Performance Contracts** (from FR-001 to FR-004):
```yaml
# performance_contracts.yaml
contracts:
  - id: PC-001
    requirement: FR-001
    description: Individual ticket processing ≤15 seconds
    test: Process single ticket with 8-12 entities
    assertion: total_time_ms <= 15000

  - id: PC-002
    requirement: FR-002
    description: Throughput ≥240 tickets/hour
    test: Process 100 continuous tickets
    assertion: tickets_per_hour >= 240

  - id: PC-003
    requirement: FR-003
    description: Storage operations ≤10 seconds
    test: Measure storage time post-extraction
    assertion: storage_time_ms <= 10000

  - id: PC-004
    requirement: FR-004
    description: Complete dataset ≤17 hours
    test: Extrapolate from 1000-ticket run
    assertion: estimated_total_hours <= 17
```

**Data Integrity Contracts** (from FR-005 to FR-007):
```yaml
# data_integrity_contracts.yaml
contracts:
  - id: DIC-001
    requirement: FR-005
    description: 100% data integrity maintained
    test: Process 50 tickets, verify no entity loss
    assertion: extracted_entity_count == stored_entity_count

  - id: DIC-002
    requirement: FR-006
    description: Stored entities match extracted entities
    test: Compare entity content extracted vs stored
    assertion: entity_content_hash_extracted == entity_content_hash_stored

  - id: DIC-003
    requirement: FR-007
    description: All relationships preserved
    test: Verify relationship integrity after storage
    assertion: relationship_count_extracted == relationship_count_stored AND all_foreign_keys_valid
```

**Monitoring Contracts** (from FR-008 to FR-011):
```yaml
# monitoring_contracts.yaml
contracts:
  - id: MC-001
    requirement: FR-008
    description: Track processing time with millisecond precision
    test: Verify ProcessingMetrics records have ms precision
    assertion: metrics.timestamp_precision == 'millisecond'

  - id: MC-002
    requirement: FR-009
    description: Track throughput in real-time
    test: Query throughput metric during processing
    assertion: throughput_metric_exists AND throughput_updated_realtime

  - id: MC-003
    requirement: FR-010
    description: Alert when processing >20 seconds per ticket
    test: Simulate slow ticket, verify alert triggered
    assertion: alert_triggered_when(total_time_ms > 20000)

  - id: MC-004
    requirement: FR-011
    description: Log timing breakdowns
    test: Verify logs contain extraction_time and storage_time
    assertion: 'extraction_time' in log_entry AND 'storage_time' in log_entry
```

### 3. Generate Contract Tests

**test_performance_contract.py**:
- `test_pc001_single_ticket_15_seconds()` - MUST fail initially
- `test_pc002_throughput_240_per_hour()` - MUST fail initially
- `test_pc003_storage_10_seconds()` - MUST fail initially
- `test_pc004_dataset_17_hours()` - MUST fail initially

**test_data_integrity_contract.py**:
- `test_dic001_no_entity_loss()` - MUST pass (no data corruption)
- `test_dic002_entity_content_match()` - MUST pass (exact content)
- `test_dic003_relationship_integrity()` - MUST pass (all relationships preserved)

**test_monitoring_contract.py**:
- `test_mc001_millisecond_precision()` - MUST fail initially (not implemented)
- `test_mc002_realtime_throughput()` - MUST fail initially (not implemented)
- `test_mc003_slow_ticket_alert()` - MUST fail initially (not implemented)
- `test_mc004_timing_breakdowns()` - MUST fail initially (not implemented)

### 4. Extract Test Scenarios from User Stories

**Primary User Story**: Ticket ingestion system operator needs system to process tickets within expected timeframes (10-15 seconds each).

**Integration Test Scenarios**:

1. **Scenario: Process single ticket with typical entity count**
   - Given: Fresh IRIS database with no entities
   - When: Process one ticket with 10 entities and 5 relationships
   - Then: Total time ≤15 seconds AND all entities stored AND all relationships valid

2. **Scenario: Process batch of 100 tickets continuously**
   - Given: Fresh IRIS database
   - When: Process 100 tickets with 8-12 entities each
   - Then: Throughput ≥240/hour AND memory usage stable AND all data validated

3. **Scenario: Complete dataset processing within time limit**
   - Given: 10,150 tickets ready for processing
   - When: Process all tickets with optimized pipeline
   - Then: Total time ≤17 hours AND 100% data integrity validated

4. **Scenario: Extraction fast, storage optimized**
   - Given: Single ticket ready for processing
   - When: Extraction completes in 5-6 seconds
   - Then: Storage completes in ≤10 seconds (not 50-120 seconds)

### 5. Update Agent Context (CLAUDE.md)

Run `.specify/scripts/bash/update-agent-context.sh claude` to add:
- New technologies: Connection pooling patterns, batch entity processing
- Recent changes: GraphRAG storage optimization, performance monitoring
- Testing patterns: Performance contract tests with throughput validation

**Output**:
- data-model.md with Ticket, Entity, Relationship, ProcessingMetrics entities
- contracts/performance_contracts.yaml, data_integrity_contracts.yaml, monitoring_contracts.yaml
- Failing contract tests in tests/contract/
- quickstart.md with performance validation workflow
- CLAUDE.md updated with new context

## Phase 1 Constitution Re-Check

**Post-Design Constitution Review**:

**I. Framework-First Architecture**: ✓ All components remain framework-level, no application logic
**II. Pipeline Validation & Requirements**: ✓ Existing validation untouched
**III. Test-Driven Development**: ✓ Contract tests written, will fail initially (TDD principle)
**IV. Performance & Enterprise Scale**: ✓ Addresses 10K+ document performance (core goal)
**V. Production Readiness**: ✓ Monitoring and logging enhanced
**VI. Explicit Error Handling**: ✓ All storage errors will surface with context
**VII. Standardized Database Interfaces**: ✓ Uses existing utilities, contributes batching patterns back

**Post-Design Constitution Check: PASS** ✅

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Load `.specify/templates/tasks-template.md` as base structure
2. Generate tasks from Phase 1 contracts and data model:
   - Each performance contract → performance contract test task [P]
   - Each data integrity contract → data integrity contract test task [P]
   - Each monitoring contract → monitoring contract test task [P]
   - Data model entities → no new models (existing IRIS tables)
   - Unified embedding service → integration task
   - Batch entity processor → implementation task
   - Connection pooling → implementation task
   - Performance monitoring → implementation task
3. Order by TDD principles: Tests first, then implementation
4. Mark [P] for parallel-executable tasks (independent test files)

**Ordering Strategy**:
1. **Phase 0**: Research complete (already done via performance issue report)
2. **Phase 1**: Contract test creation (all parallel [P])
   - Write performance contract tests
   - Write data integrity contract tests
   - Write monitoring contract tests
3. **Phase 2**: Core optimization implementation
   - Task 1: Integrate UnifiedEmbeddingService into all services
   - Task 2: Implement batch_entity_processor with batching logic
   - Task 3: Add connection pooling to iris_connection_manager
   - Task 4: Implement performance_monitor for real-time tracking
4. **Phase 3**: Integration tests (parallel where possible)
   - Test unified embedding integration
   - Test batch entity storage
   - Test connection pooling
   - Test performance monitoring
5. **Phase 4**: Performance validation
   - Run 100-ticket throughput test
   - Run 1,000-ticket sustained load test
   - Validate all contract tests pass

**Estimated Output**: 18-22 numbered, dependency-ordered tasks in tasks.md

**Task Priorities**:
- P0: Contract test creation (gates implementation)
- P0: UnifiedEmbeddingService integration (12-30 sec savings per ticket)
- P0: Batch entity storage (30-64 sec savings per ticket)
- P1: Connection pooling (3-7 sec savings per ticket)
- P1: Performance monitoring (observability for validation)

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation following constitutional TDD principles
**Phase 5**: Validation against all acceptance criteria from spec.md

**Validation Criteria**:
- All 14 functional requirements (FR-001 to FR-014) verified
- Performance targets achieved: 10-15s/ticket, 240-360/hour, 11-17 hours total
- Data integrity: 100% entity preservation, 100% relationship preservation
- Monitoring: Real-time throughput tracking, alert on >20s tickets
- Operational: Graceful shutdown, resume from last completed

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

No constitutional violations. Feature aligns with all constitutional principles:
- Framework-first: Pure framework enhancement
- Performance focus: Directly addresses Principle IV (Enterprise Scale)
- TDD approach: Contract tests before implementation
- Production readiness: Enhances existing monitoring and logging

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) ✅
- [x] Phase 1: Design complete (/plan command) ✅
- [x] Phase 2: Task planning complete (/plan command - describe approach only) ✅
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS ✅
- [x] Post-Design Constitution Check: PASS ✅
- [x] All NEEDS CLARIFICATION resolved (via performance issue report)
- [x] Complexity deviations documented (none - all principles aligned)

**Artifacts Generated**:
- [x] research.md (Phase 0)
- [x] data-model.md (Phase 1)
- [x] contracts/performance_contracts.yaml (Phase 1)
- [x] contracts/data_integrity_contracts.yaml (Phase 1)
- [x] contracts/monitoring_contracts.yaml (Phase 1)
- [x] quickstart.md (Phase 1)
- [x] CLAUDE.md updated (Phase 1)

---
*Based on Constitution v1.8.0 - See `/.specify/memory/constitution.md`*
