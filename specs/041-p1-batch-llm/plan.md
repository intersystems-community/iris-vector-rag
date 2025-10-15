# Implementation Plan: Batch LLM Entity Extraction Integration

**Branch**: `041-p1-batch-llm` | **Date**: 2025-10-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/041-p1-batch-llm/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✓ COMPLETE: Spec loaded and analyzed
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → ✓ COMPLETE: All clarifications resolved (5 Q&A pairs in spec)
3. Fill the Constitution Check section
   → ✓ COMPLETE: All 7 principles verified
4. Evaluate Constitution Check section
   → ✓ COMPLETE: PASS (no violations detected)
5. Execute Phase 0 → research.md
   → ✓ COMPLETE: research.md generated
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   → ✓ COMPLETE: All Phase 1 outputs generated
7. Re-evaluate Constitution Check
   → ✓ COMPLETE: PASS (no violations detected)
8. Plan Phase 2 → Describe task generation approach
   → ✓ COMPLETE: Task planning approach documented
9. STOP - Ready for /tasks command
   → ✓ COMPLETE: Feature ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 8. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Integrate the existing BatchEntityExtractionModule (`iris_rag/dspy_modules/batch_entity_extraction.py`) into the main entity extraction pipeline to achieve 3x speedup (7.7 hours → 2.5 hours) when processing large document collections (8,000+ documents). The system will dynamically batch documents based on token count (up to 8K tokens per batch), retry failed batches with exponential backoff, and maintain current extraction quality (4.86 entities per document average). This is a P1 performance optimization following successful P0 fixes (connection pooling and JSON parsing retry).

## Technical Context
**Language/Version**: Python 3.11 (matching existing rag-templates framework)
**Primary Dependencies**: DSPy (existing), tiktoken (token counting), existing iris_rag framework
**Storage**: IRIS database (existing vector store and entity storage)
**Testing**: pytest with IRIS database integration (per constitution)
**Target Platform**: Linux server / macOS development
**Project Type**: Single project (rag-templates framework extension)
**Performance Goals**: 3x speedup (8.33 tickets/min → 25 tickets/min), maintain 4.86 entities/doc quality
**Constraints**: 8,192 token budget per batch (default, configurable), exponential backoff (2s, 4s, 8s), max 3 retries
**Scale/Scope**: Tested with 8,051 TrakCare tickets, designed for 10K+ document collections

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component extends existing EntityExtractionService (no new pipeline needed)
- ✓ No application-specific logic (generic batch processing for any document type)
- ✓ CLI interface exposed via existing pipeline orchestration

**II. Pipeline Validation & Requirements**:
- ✓ Automated requirement validation included (batch size validation, token budget checks)
- ✓ Setup procedures idempotent (batch processing can be enabled/disabled without state corruption)

**III. Test-Driven Development**:
- ✓ Contract tests written before implementation (Phase 1 output)
- ✓ Performance tests for 1K and 10K document scenarios (validating 3x speedup requirement)
- ✓ IRIS database required for all integration/e2e tests

**IV. Performance & Enterprise Scale**:
- ✓ Incremental indexing supported (batch queue allows continuous processing)
- ✓ IRIS vector operations optimized (existing infrastructure reused)
- ✓ Memory usage monitored (token budget prevents context overflow)

**V. Production Readiness**:
- ✓ Structured logging included (batch statistics, retry attempts, failure tracking)
- ✓ Health checks implemented (batch processing success rate monitoring)
- ✓ Docker deployment ready (existing framework Docker infrastructure)

**VI. Explicit Error Handling**:
- ✓ No silent failures (exponential backoff + batch splitting on persistent failures)
- ✓ Clear exception messages (batch failure context, document traceability preserved)
- ✓ Actionable error context (FR-007 processing statistics for debugging)

**VII. Standardized Database Interfaces**:
- ✓ Uses proven EntityStorageAdapter (existing framework component)
- ✓ No ad-hoc IRIS queries (reuses existing entity storage patterns)
- ✓ New patterns (batch metrics tracking) will be contributed back to framework

**Initial Constitution Check**: ✅ PASS (no violations detected)

## Project Structure

### Documentation (this feature)
```
specs/041-p1-batch-llm/
├── plan.md              # This file (/plan command output)
├── spec.md              # Feature specification (completed with clarifications)
├── research.md          # Phase 0 output (/plan command) - PENDING
├── data-model.md        # Phase 1 output (/plan command) - PENDING
├── quickstart.md        # Phase 1 output (/plan command) - PENDING
├── contracts/           # Phase 1 output (/plan command) - PENDING
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_rag/
├── services/
│   └── entity_extraction.py          # Add batch processing integration
├── dspy_modules/
│   ├── batch_entity_extraction.py     # Existing module (already implemented)
│   └── entity_extraction_module.py    # Existing single-doc module
├── core/
│   └── models.py                      # Add DocumentBatch, BatchExtractionResult models
└── utils/
    └── token_counter.py               # NEW: Token counting utility

common/
└── batch_utils.py                     # NEW: Batch queue management, retry logic

tests/
├── contract/
│   ├── test_batch_extraction_contract.py     # NEW: Batch API contract tests
│   └── test_batch_metrics_contract.py        # NEW: Metrics API contract tests
├── integration/
│   ├── test_batch_extraction_e2e.py          # NEW: End-to-end batch processing
│   ├── test_batch_retry_logic.py             # NEW: Exponential backoff validation
│   └── test_batch_performance.py             # NEW: 3x speedup validation (1K, 10K docs)
└── unit/
    ├── test_token_counter.py                  # NEW: Token counting logic
    ├── test_batch_queue.py                    # NEW: Batch queue management
    └── test_batch_sizing.py                   # NEW: Dynamic batch sizing

config/
└── memory_config.yaml                 # Update: Add batch_processing section
```

**Structure Decision**: Single project structure extending existing rag-templates framework. New components integrate into existing `iris_rag/` package structure, following established patterns for services, models, and utilities. Tests follow existing three-tier structure (contract/integration/unit).

## Phase 0: Outline & Research

**No unknowns remain** - All NEEDS CLARIFICATION markers were resolved during /clarify phase:
1. Batch failure recovery → Exponential backoff + batch splitting (clarified)
2. Batch sizing strategy → Dynamic by token count (clarified)
3. Single-document processing → Always batch (clarified)
4. Empty extraction results → Continue normally (clarified)
5. Document ordering → Reordering acceptable (clarified)

**Research tasks** (to validate assumptions and find best practices):

1. **Token Counting for Batch Sizing**
   - Research: Best practices for estimating LLM token usage from document text
   - Decision target: tiktoken library vs. custom approximation
   - Rationale needed: Accuracy vs. performance tradeoff

2. **Batch Queue Management**
   - Research: Python queue patterns for dynamic batching with token budgets
   - Decision target: Queue implementation (collections.deque, asyncio.Queue, custom)
   - Rationale needed: Thread safety, performance, integration with existing pipeline

3. **Exponential Backoff Implementation**
   - Research: Retry patterns in Python (tenacity, backoff, custom)
   - Decision target: Library vs. custom implementation
   - Rationale needed: Simplicity vs. flexibility

4. **DSPy Batch Module Integration**
   - Research: Review existing BatchEntityExtractionModule implementation
   - Decision target: Modifications needed for production use
   - Rationale needed: Validate JSON parsing robustness (related to previous 0.7% failure fix)

**Output**: research.md with consolidated findings (generated next)

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

**Data Model Extraction** (from spec Key Entities):

1. **DocumentBatch** (new model in `iris_rag/core/models.py`):
   - batch_id: str (UUID)
   - document_ids: List[str]
   - batch_size: int (variable, 1-10+)
   - total_token_count: int
   - creation_timestamp: datetime
   - processing_status: Enum (PENDING, PROCESSING, COMPLETED, FAILED)

2. **BatchExtractionResult** (new model):
   - batch_id: str (foreign key to DocumentBatch)
   - per_document_entities: Dict[str, List[Entity]]
   - per_document_relationships: Dict[str, List[Relationship]]
   - processing_time: float (seconds)
   - success_status: bool
   - retry_count: int

3. **ProcessingMetrics** (new model for statistics):
   - total_batches_processed: int
   - total_documents_processed: int
   - average_batch_processing_time: float
   - speedup_factor: float (vs. single-document baseline)
   - entity_extraction_rate_per_batch: float
   - zero_entity_documents_count: int

**API Contracts** (from Functional Requirements):

1. **Batch Processing API** (internal service interface):
   ```python
   # Contract: EntityExtractionService.extract_batch()
   def extract_batch(documents: List[Document]) -> BatchExtractionResult:
       """
       Process multiple documents in single LLM call.

       Requirements:
       - FR-001: Process 5-10 documents per batch
       - FR-006: Respect 8K token budget
       - FR-005: Retry with exponential backoff on failure
       """
   ```

2. **Token Counting API**:
   ```python
   # Contract: TokenCounter.estimate_tokens()
   def estimate_tokens(text: str) -> int:
       """Estimate token count for batch sizing."""
   ```

3. **Batch Queue API**:
   ```python
   # Contract: BatchQueue.add_document() / get_next_batch()
   def add_document(document: Document) -> None:
       """Add document to batch queue."""

   def get_next_batch(token_budget: int = 8192) -> List[Document]:
       """Get next batch of documents up to token budget."""
   ```

4. **Metrics API** (FR-007):
   ```python
   # Contract: BatchMetricsTracker.get_statistics()
   def get_statistics() -> ProcessingMetrics:
       """Return current batch processing statistics."""
   ```

**Contract Test Generation**:
- `tests/contract/test_batch_extraction_contract.py`: Validates extract_batch() signature, return type, error handling
- `tests/contract/test_batch_metrics_contract.py`: Validates metrics API returns required fields
- `tests/contract/test_token_counter_contract.py`: Validates token estimation accuracy (±10% tolerance)

**Integration Test Scenarios** (from User Stories):
- Scenario 1 (AS-1): 1,000 documents → 3x speedup validation
- Scenario 2 (AS-2): Batch failure → retry with exponential backoff
- Scenario 3 (AS-3): Entity traceability → document ID preservation
- Scenario 4 (AS-4): Variable document sizes → dynamic batch sizing
- Scenario 5 (AS-5): Single document → batch queue integration

**Agent File Update**:
- Run: `.specify/scripts/bash/update-agent-context.sh claude`
- Add: Batch processing patterns, token counting, exponential backoff strategy
- Keep: Existing RAG framework context, DSPy patterns

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Load `.specify/templates/tasks-template.md` as base
2. Generate tasks from Phase 1 design docs:
   - Each contract → contract test task [P]
   - Each model (DocumentBatch, BatchExtractionResult, ProcessingMetrics) → implementation task [P]
   - Each API (extract_batch, estimate_tokens, get_next_batch) → implementation task
   - Each user story → integration test task
   - Performance validation → 3x speedup test task

**Ordering Strategy** (TDD order):
1. Contract tests (all parallel) [P]
2. Token counting utility (foundational)
3. Batch queue management (depends on token counter)
4. Batch extraction service integration (depends on queue)
5. Retry logic with exponential backoff (depends on batch service)
6. Metrics tracking (depends on batch service)
7. Integration tests (depends on implementation)
8. Performance tests (final validation)

**Parallelization Opportunities** [P]:
- Contract test writing (independent)
- Model creation (independent files)
- Unit test implementation (independent components)

**Estimated Output**: ~20-25 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation with 1K and 10K document sets)

## Complexity Tracking
*No constitutional violations - no entries required*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - research.md generated
- [x] Phase 1: Design complete (/plan command) - data-model.md, contracts/, quickstart.md, CLAUDE.md updated
- [x] Phase 2: Task planning complete (/plan command - approach described in plan.md)
- [x] Phase 3: Tasks generated (/tasks command) - 25 tasks in tasks.md, ready for execution
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS (no violations detected)
- [x] All NEEDS CLARIFICATION resolved (5 Q&A pairs in spec)
- [x] Complexity deviations documented (none required - no violations)

---
*Based on Constitution v1.6.0 - See `/.specify/memory/constitution.md`*
