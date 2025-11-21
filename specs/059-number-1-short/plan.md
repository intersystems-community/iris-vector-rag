# Implementation Plan: Batch Storage Optimization

**Branch**: `059-number-1-short` | **Date**: 2025-01-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/059-number-1-short/spec.md`

## Summary

Optimize entity storage operations in GraphRAG pipelines by replacing inefficient individual INSERT loops with batch operations. Current implementation in `storage.py:store_entities_batch()` calls `store_entity()` individually for each entity, resulting in 200 separate database round-trips for a 20-document workload. Solution implements two-phase optimization: Phase 1 accumulates entities across documents (3-5x improvement), Phase 2 utilizes existing ConnectionManager `execute_many()` capability for true batch INSERTs (additional 2-10x improvement), achieving target throughput of 2-3 docs/sec from baseline 0.21 docs/sec.

## Technical Context

**Language/Version**: Python 3.10+ (existing iris-vector-rag requirement)
**Primary Dependencies**: iris-vector-rag framework, intersystems-irispython>=5.1.2, iris.dbapi
**Storage**: InterSystems IRIS database with vector search capabilities
**Testing**: pytest with contract tests, integration tests (real IRIS), performance benchmarks
**Target Platform**: Linux/macOS server environments, Docker deployments
**Project Type**: Single project (RAG framework enhancement)
**Performance Goals**: 10-15x throughput improvement (0.21 → 2-3 docs/sec), 200→1 INSERT reduction
**Constraints**: <30 min Phase 1 implementation, <2 hours Phase 2, maintain API compatibility, no schema changes
**Scale/Scope**: 200-10,000 entities per batch, 1000+ document collections

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Enhances existing StorageService framework component
- ✓ No application-specific logic (pure storage optimization)
- ✓ Already exposed via CLI through pipeline operations

**II. Pipeline Validation & Requirements**:
- ✓ Maintains existing validation logic
- ✓ Setup procedures remain idempotent (batch operations transparent to callers)

**III. Test-Driven Development**:
- ✓ Contract tests required for batch operations before implementation
- ✓ Performance tests with 10K+ entity workload (measure baseline vs optimized)
- ✓ Integration tests with real IRIS database using iris-devtester

**IV. Performance & Enterprise Scale**:
- ✓ Core optimization FOR enterprise scale
- ✓ Directly optimizes IRIS vector operations (batch INSERT efficiency)
- ✓ Incremental processing maintained (batch size configurable)

**V. Production Readiness**:
- ✓ Structured logging for batch metrics (entities/time/success rate)
- ✓ Health checks inherited from existing framework
- ✓ Docker deployment unchanged (framework enhancement)

**VI. Explicit Error Handling**:
- ✓ Partial batch failures identified with clear entity-level reporting
- ✓ Graceful degradation to individual INSERT on batch failure
- ✓ No silent failures (all errors surfaced with context)

**VII. Standardized Database Interfaces**:
- ✓ Uses existing ConnectionManager.execute_many() utility
- ✓ No ad-hoc IRIS queries (follows established patterns)
- ✓ Enhancement contributes back to framework (shared storage utility)

**Constitution Compliance**: ✅ **ALL GATES PASS** - No violations

## Project Structure

### Documentation (this feature)
```
specs/059-number-1-short/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_vector_rag/
├── services/
│   └── storage.py                # MODIFY: Batch optimization implementation
├── common/
│   ├── connection_manager.py     # READ: execute_many() usage patterns
│   └── connection_pool.py        # READ: Batch operation support
└── storage/
    └── schema_manager.py         # READ: Understanding storage patterns

tests/
├── contract/
│   └── test_batch_storage.py    # NEW: Batch operation contracts
├── integration/
│   └── test_batch_performance.py # NEW: Real IRIS batch performance
└── unit/
    └── test_storage_batch.py    # NEW: Batch logic unit tests
```

**Structure Decision**: Single project enhancement to existing iris-vector-rag framework. Primary modification in `iris_vector_rag/services/storage.py` (StorageService class), leveraging existing `common/connection_manager.py` execute_many() capability. All changes maintain framework architecture and backward compatibility.

## Phase 0: Outline & Research

### Research Tasks Identified

**R1: ConnectionManager execute_many() API**
- **Unknown**: Exact signature, parameter format, error handling for execute_many()
- **Task**: Read `iris_vector_rag/common/connection_manager.py` to document execute_many() usage
- **Output**: API signature, parameter requirements, exception handling patterns

**R2: Entity Storage SQL Patterns**
- **Unknown**: Current INSERT statement structure, field mapping, validation logic
- **Task**: Analyze `iris_vector_rag/services/storage.py:store_entity()` implementation
- **Output**: SQL template, field list, preparation requirements for batch execution

**R3: IRIS Batch Operation Limits**
- **Unknown**: Maximum batch size, transaction limits, memory constraints
- **Task**: Research IRIS database batch INSERT capabilities and best practices
- **Output**: Recommended batch sizes, chunking strategy, error recovery patterns

**R4: Existing Batch Accumulation Patterns**
- **Unknown**: Current batch accumulation logic (if any), memory management
- **Task**: Search codebase for existing batch patterns in storage/pipeline code
- **Output**: Reusable patterns, anti-patterns to avoid, memory management approaches

**Output**: `research.md` with all decisions documented

## Phase 1: Design & Contracts

*Prerequisites: research.md complete*

### 1. Data Model (`data-model.md`)

**Key Entities**:
- **EntityBatch**: Collection for accumulating entities before storage
- **BatchMetrics**: Tracking for batch operations (count, time, success/failure)
- **StorageOperation**: Wrapper for individual vs batch storage modes

### 2. API Contracts (`contracts/`)

**Storage Service Enhancements**:
- `store_entities_batch_optimized(entities: List[Entity]) → BatchResult`
- `configure_batch_size(max_size: int) → None`
- `get_batch_metrics() → BatchMetrics`

**Contract Tests** (must fail initially):
- `test_batch_accumulation_across_documents()`
- `test_execute_many_integration()`
- `test_batch_error_handling()`
- `test_graceful_degradation()`

### 3. Agent Context Update

Execute: `.specify/scripts/bash/update-agent-context.sh claude`

**Technologies to Add**:
- Batch storage patterns for IRIS
- execute_many() usage in ConnectionManager
- Performance benchmarking approaches

### 4. Quick Start (`quickstart.md`)

**Test Scenario**: Index 20 documents with GraphRAG, measure throughput improvement from baseline (0.21 docs/sec) through Phase 1 (0.6-1.0 docs/sec) to Phase 2 (2-3 docs/sec).

**Output**: data-model.md, contracts/, failing tests, quickstart.md, CLAUDE.md updated

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. **Contract Tests First** (TDD):
   - Task 1: Write contract test for batch accumulation [P]
   - Task 2: Write contract test for execute_many() integration [P]
   - Task 3: Write contract test for error handling [P]

2. **Phase 1 Implementation** (Cross-document accumulation):
   - Task 4: Implement EntityBatch accumulator class
   - Task 5: Modify store_entities_batch() to use accumulator
   - Task 6: Add batch size configuration
   - Task 7: Verify Phase 1 performance (3-5x improvement)

3. **Phase 2 Implementation** (Batch INSERT optimization):
   - Task 8: Refactor entity INSERT preparation
   - Task 9: Integrate execute_many() from ConnectionManager
   - Task 10: Implement batch chunking for large batches
   - Task 11: Add partial failure error handling
   - Task 12: Verify Phase 2 performance (2-10x additional improvement)

4. **Integration & Validation**:
   - Task 13: Integration tests with real IRIS [requires_database]
   - Task 14: Performance benchmark (1000 doc workload)
   - Task 15: Backward compatibility validation

**Ordering Strategy**:
- TDD order: Contract tests (Tasks 1-3) before implementation
- Phased approach: Phase 1 complete before Phase 2 start
- Dependencies: Accumulator (Task 4) before storage modification (Task 5)
- Mark [P] for parallel-safe tasks (independent test files)

**Estimated Output**: ~15 numbered, dependency-ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following TDD principles)
**Phase 5**: Validation (run all tests, execute quickstart.md performance benchmark, verify 10-15x throughput improvement)

## Complexity Tracking

*No constitutional violations - this section remains empty*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | N/A | N/A |

## Progress Tracking

*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete ✅ **CRITICAL FINDING: Feature already implemented in Feature 057**
- [ ] Phase 1: Design complete (BLOCKED - awaiting stakeholder decision on duplicate)
- [ ] Phase 2: Task planning complete (BLOCKED - awaiting stakeholder decision)
- [ ] Phase 3: Tasks generated (BLOCKED - may close as duplicate)
- [ ] Phase 4: Implementation complete (BLOCKED - implementation already exists)
- [ ] Phase 5: Validation passed (BLOCKED - validation complete in Feature 057)

**Gate Status**:
- [x] Initial Constitution Check: PASS (all gates green)
- [ ] Post-Design Constitution Check: N/A (duplicate feature - no new design)
- [x] All NEEDS CLARIFICATION resolved ✅ Research shows Feature 057 already implements this
- [x] Complexity deviations documented (none - no violations)

**Critical Research Finding**:
- ✅ BatchEntityProcessor already implements executemany() optimization (Feature 057)
- ✅ Performance targets already met (5-10x improvement, 30-64s saved per ticket)
- ✅ All FR-006 through FR-015 requirements already implemented
- ⚠️ **RECOMMENDATION: Close Feature 059 as duplicate of Feature 057**

---
*Based on Constitution v1.8.0 - See `/.specify/memory/constitution.md`*
