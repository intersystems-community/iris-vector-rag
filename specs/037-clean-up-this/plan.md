
# Implementation Plan: Repository Cleanup and Organization

**Branch**: `037-clean-up-this` | **Date**: 2025-10-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/037-clean-up-this/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✅
   → Spec found and analyzed
2. Fill Technical Context (scan for NEEDS CLARIFICATION) ✅
   → Project Type: Single Python project (repository maintenance)
   → Structure Decision: N/A (maintenance task, not new code)
3. Fill the Constitution Check section ✅
   → Repository cleanup is a maintenance task exempt from RAG pipeline requirements
4. Evaluate Constitution Check section ✅
   → PASS with N/A exemptions (not a RAG pipeline feature)
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md ✅
   → No technical research needed (file system operations)
6. Execute Phase 1 → contracts, data-model.md, quickstart.md ✅
7. Re-evaluate Constitution Check section ✅
   → PASS (no design changes)
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach ✅
9. STOP - Ready for /tasks command ✅
```

**IMPORTANT**: The /plan command STOPS at step 9. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Clean up and reorganize the rag-templates repository to remove unnecessary files, consolidate duplicates, and improve overall organization while maintaining 100% test pass rate. Primary actions include:
- Remove temporary/cache files and old evaluation reports
- Delete historical status tracking files (old session notes, completion summaries)
- Move current status files (docs/docs/STATUS.md, docs/docs/PROGRESS.md, docs/docs/TODO.md, docs/docs/docs/CHANGELOG.md) to docs/
- Consolidate duplicate documentation (keep newest, delete older)
- Validate all tests still pass after cleanup

## Technical Context

**Language/Version**: Python 3.11 (existing project)
**Primary Dependencies**: File system operations, git, pytest (for test validation)
**Storage**: File system only (no database operations)
**Testing**: pytest (validation that cleanup doesn't break tests)
**Target Platform**: macOS/Linux (development environment)
**Project Type**: Single Python project (repository maintenance task)
**Performance Goals**: N/A (one-time cleanup operation)
**Constraints**: 100% test pass rate must be maintained after cleanup
**Scale/Scope**: ~200-300 files in repository, 136 tests must continue passing

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Exemption Rationale**: Repository cleanup is a maintenance task, not a RAG pipeline implementation. Constitutional requirements (I-VII) apply to pipeline development, not file organization tasks.

**I. Framework-First Architecture**: N/A - Maintenance task, no new components
**II. Pipeline Validation & Requirements**: N/A - No pipeline being created
**III. Test-Driven Development**: ✓ Test validation required (FR-009, FR-010) - all 136 tests must pass post-cleanup
**IV. Performance & Enterprise Scale**: N/A - One-time cleanup operation
**V. Production Readiness**: N/A - Maintenance task
**VI. Explicit Error Handling**: ✓ Rollback required if tests fail (FR-011)
**VII. Standardized Database Interfaces**: N/A - No database operations

**Applicable Requirements**:
- Test validation before and after cleanup (Constitutional Principle III)
- Explicit error handling and rollback on failure (Constitutional Principle VI)
- Git history for reversibility (Constitutional governance - all changes tracked)

## Project Structure

### Documentation (this feature)
```
specs/037-clean-up-this/
├── spec.md              # Feature specification
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (minimal - no technical research needed)
├── data-model.md        # Phase 1 output (file classification model)
├── quickstart.md        # Phase 1 output (cleanup execution steps)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Repository Structure (Current State Analysis)
```
/Users/tdyar/ws/rag-templates/
├── README.md                    # Essential - keep
├── USER_GUIDE.md                # Essential - keep
├── CLAUDE.md                    # Essential - keep
├── DOCUMENTATION_INDEX.md       # Essential - keep
├── DOCUMENTATION_AUDIT_REPORT.md # Keep - recent audit
├── docs/docs/STATUS.md                    # Move to docs/
├── docs/docs/PROGRESS.md                  # Move to docs/
├── docs/docs/TODO.md                      # Move to docs/
├── docs/docs/docs/CHANGELOG.md                 # Move to docs/
├── MORNING_BRIEFING.md          # Historical - delete
├── docker-compose.yml           # Essential - keep
├── Makefile                     # Essential - keep
├── pyproject.toml               # Essential - keep
├── pytest.ini                   # Essential - keep
├── .gitignore                   # Essential - keep
├── outputs/                     # Contains old RAGAS reports - remove old files
│   ├── pipeline_verification_*.json  # Old outputs - delete
│   └── reports/ragas_evaluations/    # Old reports - delete
├── iris_rag/                    # Source code - keep all
├── tests/                       # Tests - keep all
├── docs/                        # Documentation - keep, organize
├── scripts/                     # Scripts - keep all
├── specs/                       # Specifications - keep all
└── [other directories]          # Analyze and keep/organize
```

**Structure Decision**: This is a maintenance task on an existing single-project Python repository. No new source code structure needed. Focus is on file organization and cleanup within existing structure.

## Phase 0: Outline & Research

**Research Status**: Minimal research required - this is a file organization task with clear requirements.

### Research Items

1. **Python Project Structure Standards**:
   - Decision: Follow existing flat-layout structure (iris_rag/ at root)
   - Rationale: Project already established with this pattern; no migration needed
   - Alternatives considered: src-layout would require moving iris_rag/ to src/iris_rag/, breaking existing imports

2. **File Classification Strategy**:
   - Decision: Three-category classification (Essential, Relocatable, Removable)
   - Rationale: Maps directly to FR-002 requirements
   - Implementation: Pattern matching + manual review for edge cases

3. **Test Validation Approach**:
   - Decision: Run full pytest suite before and after each file operation batch
   - Rationale: Ensures FR-009/FR-010 compliance with immediate rollback capability
   - Implementation: pytest with coverage tracking, git staging for rollback

**Output**: research.md created

## Phase 1: Design & Contracts

*Prerequisites: research.md complete*

### Data Model

**File Classification Model** (see data-model.md for full schema):
- TopLevelFile (path, category: Essential|Relocatable|Removable, reason)
- DocumentationFile (path, purpose, target_location, is_duplicate)
- GeneratedOutput (path, generation_date, is_outdated)
- StatusFile (path, is_current, target_location)

### Cleanup Operations Contract

**Operations** (see contracts/cleanup-operations.md):
1. `scan_repository()` → FileInventory
2. `classify_files(inventory)` → ClassifiedFiles
3. `remove_files(removable_files)` → RemovalReport
4. `move_files(relocatable_files, target_map)` → MoveReport
5. `validate_tests()` → TestReport
6. `rollback_changes()` → RollbackReport
7. `update_documentation(moved_files)` → UpdateReport

### Contract Tests

Test files created in `tests/contract/`:
- `test_file_classification_contract.py` - Validates file categorization logic
- `test_cleanup_operations_contract.py` - Validates cleanup operation contracts
- `test_test_validation_contract.py` - Validates test suite execution and comparison
- `test_documentation_update_contract.py` - Validates doc link updates

### Quickstart Execution

See `quickstart.md` for step-by-step cleanup execution:
1. Create backup branch
2. Run initial test suite
3. Scan and classify files
4. Execute cleanup in batches (remove → move → update docs)
5. Validate tests after each batch
6. Update DOCUMENTATION_INDEX.md
7. Commit and verify

### Agent Context Update

Running incremental update: `.specify/scripts/bash/update-agent-context.sh claude`

**Output**: data-model.md, contracts/, test stubs, quickstart.md, CLAUDE.md updated

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from cleanup operations contract
- Each classification rule → validation task
- Each cleanup operation → implementation + test task
- Documentation updates → final verification task

**Ordering Strategy**:
1. Setup tasks (backup, baseline test run) [P]
2. File classification implementation [P]
3. Contract tests for classification
4. Removal operation implementation
5. Contract tests for removal
6. Move operation implementation
7. Contract tests for move
8. Documentation update implementation
9. Contract tests for doc updates
10. Integration test (full cleanup workflow)
11. Quickstart validation

**Estimated Output**: 20-25 numbered, ordered tasks in tasks.md

**Task Categories**:
- [P] Parallel execution possible (independent file analysis)
- Sequential execution required (cleanup operations with test validation)
- Final validation tasks (integration tests, quickstart)

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following cleanup plan)
**Phase 5**: Validation (100% test pass rate, documentation links valid, no broken imports)

## Complexity Tracking

*No constitutional violations - this is a maintenance task exempt from RAG pipeline requirements*

| Requirement | Status | Notes |
|-------------|--------|-------|
| Test validation | ✓ | Required by FR-009/FR-010 and Constitutional Principle III |
| Error handling | ✓ | Rollback required by FR-011 and Constitutional Principle VI |
| Git history | ✓ | Reversibility required by NFR-001 |

## Progress Tracking

*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS (N/A exemptions documented)
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (6 of 7 critical items resolved; 1 low-priority resolved by following existing structure)
- [x] Complexity deviations documented (none - maintenance task)

---
*Based on Constitution v1.6.0 - See `/.specify/memory/constitution.md`*
