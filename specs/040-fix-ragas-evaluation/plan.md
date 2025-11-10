
# Implementation Plan: Fix RAGAS GraphRAG Evaluation Workflow

**Branch**: `040-fix-ragas-evaluation` | **Date**: 2025-10-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/intersystems-community/ws/rag-templates/specs/040-fix-ragas-evaluation/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✓
   → Loaded spec: Fix RAGAS GraphRAG evaluation workflow
2. Fill Technical Context ✓
   → No NEEDS CLARIFICATION markers found
   → Project Type: Single Python project (evaluation framework)
3. Fill the Constitution Check section ✓
   → Evaluated against all 7 constitutional principles
4. Evaluate Constitution Check section ✓
   → Minor violations documented (no new framework component needed)
   → Justification: Bug fix to existing evaluation workflow
5. Execute Phase 0 → research.md
   → Skipped (no unknowns, modifying existing script)
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
7. Re-evaluate Constitution Check section
   → No new violations introduced
8. Plan Phase 2 → Describe task generation approach
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 9. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Fix the RAGAS evaluation workflow (`scripts/simple_working_ragas.py`) to properly handle GraphRAG pipeline testing. Currently, GraphRAG fails with "Knowledge graph is empty" because the workflow assumes all pipelines work with basic document data. GraphRAG requires entity extraction to populate knowledge graph tables (RAG.Entities, RAG.EntityRelationships). The fix will detect when GraphRAG is being tested and either: (1) auto-load documents using GraphRAG's `load_documents()` method to extract entities, or (2) skip GraphRAG evaluation with a clear message explaining the missing prerequisite.

## Technical Context
**Language/Version**: Python 3.12 (existing project)
**Primary Dependencies**:
- iris_rag framework (GraphRAG pipeline, entity extraction service)
- RAGAS evaluation framework
- sentence-transformers (embedding models)
- spacy/scispacy (entity extraction - optional)
**Storage**: InterSystems IRIS database (tables: RAG.SourceDocuments, RAG.Entities, RAG.EntityRelationships)
**Testing**: pytest with IRIS database integration
**Target Platform**: Development environment (macOS/Linux with Docker IRIS)
**Project Type**: Single Python project (evaluation framework modification)
**Performance Goals**: Entity extraction for 10-71 sample documents in <30 seconds
**Constraints**:
- Must not break existing basic/crag/pylate_colbert pipeline evaluations
- Must handle cases where entity extraction fails or returns zero entities
- Must be backward compatible with existing RAGAS evaluation targets
**Scale/Scope**:
- 1 script to modify (scripts/simple_working_ragas.py)
- 71 sample documents for testing
- 5 pipeline types (basic, basic_rerank, crag, graphrag, pylate_colbert)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ⚠️ Not a new component - bug fix to evaluation script | ✓ Reuses existing GraphRAG pipeline | ✓ CLI interface exists (make test-ragas-sample)

**II. Pipeline Validation & Requirements**: ✓ Will validate entity data before GraphRAG evaluation | ✓ Setup procedure remains idempotent (uses existing load_documents)

**III. Test-Driven Development**: ✓ Will verify fix with existing test-ragas-sample target | ⚠️ No new contract tests needed (modifying existing script)

**IV. Performance & Enterprise Scale**: N/A - Evaluation script modification, not core framework | ✓ Uses existing IRIS vector operations

**V. Production Readiness**: ✓ Will add structured logging for GraphRAG entity check | ✓ Existing health checks remain | ✓ Docker deployment unaffected

**VI. Explicit Error Handling**: ✓ Clear messages for missing entity data | ✓ No silent failures - explicit skip or auto-load | ✓ Actionable context provided

**VII. Standardized Database Interfaces**: ✓ Uses existing GraphRAG pipeline methods | ✓ No new ad-hoc queries | ✓ Reuses proven entity extraction service

**Constitution Compliance Summary**: This is a bug-fix feature for the evaluation workflow, not a new framework component. Most principles (I, III, IV) have relaxed requirements for evaluation scripts. Key improvements: explicit error handling (VI) for missing entity data, proper use of framework methods (VII) via GraphRAG's load_documents(), and clear logging/skip messages (V). No constitutional violations that cannot be justified.

## Project Structure

### Documentation (this feature)
```
specs/040-fix-ragas-evaluation/
├── plan.md              # This file (/plan command output)
├── spec.md              # Feature specification (completed)
├── research.md          # Phase 0 output (SKIPPED - no unknowns)
├── data-model.md        # Phase 1 output (evaluation workflow states)
├── quickstart.md        # Phase 1 output (validation procedure)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
scripts/
└── simple_working_ragas.py          # Target file to modify (main evaluation script)

iris_rag/pipelines/
└── graphrag.py                      # GraphRAG pipeline with load_documents() method (existing)

iris_rag/services/
└── entity_extraction.py             # Entity extraction service (existing, used by GraphRAG)

tests/
└── (no new test files - validation via make test-ragas-sample)

data/sample_10_docs/                 # Sample documents for testing (existing)

Makefile                             # Contains test-ragas-sample target (existing)
```

**Structure Decision**: Single Python project structure. This is a modification to an existing evaluation script (`scripts/simple_working_ragas.py`). No new directories or major structural changes needed. The fix reuses existing GraphRAG pipeline infrastructure (load_documents method, entity extraction service) and database tables (RAG.Entities, RAG.EntityRelationships).

## Phase 0: Outline & Research

**SKIPPED - No research needed for this feature.**

**Rationale**: This is a bug fix to an existing evaluation script. All components already exist:
- GraphRAG pipeline with `load_documents()` method (iris_rag/pipelines/graphrag.py)
- Entity extraction service (iris_rag/services/entity_extraction.py)
- Knowledge graph tables (RAG.Entities, RAG.EntityRelationships)
- RAGAS evaluation framework integration

The fix requires no new technologies, no unknowns, and no research. It's a straightforward modification to detect missing entity data and either auto-load or skip evaluation.

**Output**: No research.md file created (skipped per execution flow)

## Phase 1: Design & Contracts
*Prerequisites: research.md complete (skipped for this feature)*

### 1. Data Model (data-model.md)

**Entity: Evaluation Pipeline State**
- pipeline_name: str (basic, basic_rerank, crag, graphrag, pylate_colbert)
- requires_entities: bool (True for graphrag, hybrid_graphrag)
- entity_check_passed: bool (True if entity data exists OR auto-load succeeded)
- evaluation_status: enum (pending, running, completed, skipped, failed)
- skip_reason: Optional[str] (e.g., "Knowledge graph empty - no entities found")

**Entity: Entity Data Check Result**
- entities_count: int (rows in RAG.Entities)
- relationships_count: int (rows in RAG.EntityRelationships)
- has_sufficient_data: bool (entities_count > 0)
- check_timestamp: datetime

**Entity: Auto-Load Result**
- documents_loaded: int
- entities_extracted: int
- relationships_extracted: int
- load_success: bool
- error_message: Optional[str]

**Workflow State Transitions**:
```
For GraphRAG pipeline:
1. pending → check entity data
2. entity_check → has_data? → running (evaluate)
             \→ no_data? → auto_load_mode? → load_documents → running
                       \→ skip_mode? → skipped (log reason)
                       \→ fail_mode? → failed (raise error)
3. running → completed (success) OR failed (error)
```

### 2. API Contracts (contracts/)

**No external APIs** - This is an internal evaluation script modification. Contracts are method signatures:

**Contract 1**: Entity data checking
```python
def check_graphrag_prerequisites() -> Dict[str, Any]:
    """
    Check if GraphRAG prerequisites are met.

    Returns:
        {
            "has_entities": bool,
            "entities_count": int,
            "relationships_count": int,
            "sufficient_data": bool
        }
    """
```

**Contract 2**: Auto-load decision
```python
def should_auto_load_entities(
    pipeline_name: str,
    entity_check: Dict[str, Any],
    config: Dict[str, Any]
) -> bool:
    """
    Determine if documents should be auto-loaded with entity extraction.

    Args:
        pipeline_name: Name of pipeline being tested
        entity_check: Result from check_graphrag_prerequisites()
        config: Evaluation configuration (mode: auto_load|skip|fail)

    Returns:
        True if auto-load should proceed, False otherwise
    """
```

**Contract 3**: GraphRAG document loading
```python
def load_documents_with_entities(
    pipeline: GraphRAGPipeline,
    documents_path: str
) -> Dict[str, Any]:
    """
    Load documents using GraphRAG pipeline to extract entities.

    Args:
        pipeline: GraphRAG pipeline instance
        documents_path: Path to documents directory

    Returns:
        {
            "documents_loaded": int,
            "entities_extracted": int,
            "relationships_extracted": int,
            "success": bool,
            "error": Optional[str]
        }
    """
```

### 3. Contract Tests (contracts/)

**No formal contract test files** for this bug fix. Validation will be done via:
- Running `make test-ragas-sample` and verifying GraphRAG no longer fails
- Checking that entity count is logged before GraphRAG evaluation
- Verifying skip message appears when entity data is missing (if skip mode)

### 4. Test Scenarios (quickstart.md)

See quickstart.md (created below)

### 5. Update Agent File

Will run `.specify/scripts/bash/update-agent-context.sh claude` to add context about GraphRAG entity extraction requirement.

**Output**:
- data-model.md (evaluation workflow states)
- contracts/ (method signatures documented above, no separate files)
- quickstart.md (validation procedure)
- CLAUDE.md (updated with new context)

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from the fix requirements (not from contracts since this is a script modification)
- Task sequence:
  1. **Baseline** [P]: Capture current GraphRAG failure state (run test-ragas-sample, verify error)
  2. **Entity Check** [P]: Add function to check RAG.Entities and RAG.EntityRelationships tables
  3. **Auto-Load Logic**: Add function to load documents via GraphRAG.load_documents()
  4. **Skip Logic**: Add logging for skip message when entity data missing
  5. **Integration**: Modify main evaluation loop to call entity check before GraphRAG
  6. **Validation**: Re-run test-ragas-sample, verify GraphRAG succeeds or skips cleanly
  7. **Documentation**: Update quickstart.md with before/after comparison

**Ordering Strategy**:
- Baseline first (validate problem exists)
- Helper functions in parallel [P] (entity check, auto-load, skip logic)
- Integration task (depends on helpers)
- Validation last (verify fix works)

**Estimated Output**: 7-8 numbered tasks in tasks.md (mix of sequential and parallel)

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run quickstart.md, verify GraphRAG evaluation works)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

**Violation**: Principles I and III have minor violations (not a framework component, no formal contract tests)

**Justification**: This is a bug fix to an evaluation script, not a new framework component. The evaluation workflow is not part of the core RAG framework - it's a testing/validation tool. Adding full contract tests for a single-script bug fix would create unnecessary overhead. The fix reuses existing framework components (GraphRAG pipeline, entity extraction service) which already have comprehensive tests.

**Alternatives Considered**:
1. **Create full contract test suite**: Rejected - overkill for single script modification
2. **Refactor evaluation into framework component**: Rejected - evaluation scripts should remain separate from core framework
3. **Current approach**: Minimal fix with validation via existing `make test-ragas-sample` target

**Mitigation**: Validation via quickstart.md procedure ensures fix works correctly without formal contract tests.

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - SKIPPED (no unknowns)
- [x] Phase 1: Design complete (/plan command)
  - ✓ data-model.md created (evaluation workflow states)
  - ✓ quickstart.md created (validation procedure)
  - ✓ CLAUDE.md updated (GraphRAG entity extraction context)
- [x] Phase 2: Task planning approach described (/plan command)
- [x] Phase 3: Tasks generated (/tasks command)
  - ✓ tasks.md created (8 sequential tasks: T001-T008)
  - ✓ Baseline → Implementation → Validation flow
  - ✓ Estimated 2-3 hours completion time
- [ ] Phase 4: Implementation (execute tasks T001-T008)
- [ ] Phase 5: Validation (run quickstart.md tests)

**Gate Status**:
- [x] Initial Constitution Check: PASS (violations justified)
- [x] Post-Design Constitution Check: PASS (no new violations)
- [x] All NEEDS CLARIFICATION resolved (none existed)
- [x] Complexity deviations documented and justified

---
*Based on Constitution v1.6.0 - See `.specify/memory/constitution.md`*
