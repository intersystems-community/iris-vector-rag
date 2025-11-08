# Implementation Plan: Update iris-vector-rag to use iris-vector-graph 1.1.1

**Branch**: `053-update-to-iris` | **Date**: 2025-11-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/053-update-to-iris/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✅ Loaded from /specs/053-update-to-iris/spec.md
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → ✅ All context clear - import update in existing Python codebase
3. Fill the Constitution Check section based on the content of the constitution document.
   → ✅ Completed below
4. Evaluate Constitution Check section below
   → ✅ No violations - pure import path update
5. Execute Phase 0 → research.md
   → ✅ No unknowns - iris-vector-graph 1.1.1 structure is known
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   → ✅ Will generate contract tests for import validation
7. Re-evaluate Constitution Check section
   → ✅ No new violations - implementation is straightforward
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
   → ✅ Described below
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Update iris-vector-rag package to use iris-vector-graph 1.1.1's new module structure. The iris-vector-graph package was refactored from `iris_vector_graph_core` to `iris_vector_graph` top-level module. This feature updates all import statements in iris-vector-rag to use the new module path while maintaining backward compatibility through dependency version constraint (>= 1.1.1).

**Primary File to Update**: `iris_rag/pipelines/hybrid_graphrag_discovery.py` (lines 127-130, 166-169)
**Dependency Update**: `pyproject.toml` - change `iris-vector-graph>=2.0.0` to `iris-vector-graph>=1.1.1`

## Technical Context
**Language/Version**: Python 3.10+
**Primary Dependencies**: iris-vector-graph >= 1.1.1 (updated from 2.0.0)
**Storage**: N/A (no storage changes)
**Testing**: pytest with contract tests for import validation
**Target Platform**: Linux/macOS/Windows (cross-platform Python package)
**Project Type**: Single project (Python package)
**Performance Goals**: No performance impact - pure import path change
**Constraints**: Must work with iris-vector-graph >= 1.1.1, fail clearly for older versions
**Scale/Scope**: 4 import statements across 1 primary file + dependency update

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ✓ No new components | ✓ No application-specific logic | ✓ No CLI changes

**II. Pipeline Validation & Requirements**: ✓ Existing validation sufficient | ✓ No setup changes needed

**III. Test-Driven Development**: ✓ Contract tests for imports | N/A No performance impact (import change only)

**IV. Performance & Enterprise Scale**: ✓ No performance changes | ✓ No IRIS operations affected

**V. Production Readiness**: ✓ Existing logging covers import failures | ✓ No health check changes | ✓ No Docker changes

**VI. Explicit Error Handling**: ✓ ImportError raised if package not found | ✓ Clear error messages already exist | ✓ Existing error context preserved

**VII. Standardized Database Interfaces**: ✓ No database changes | ✓ No IRIS query changes | N/A No new patterns

**Constitution Compliance**: ✅ PASS - This is a pure dependency update with no architectural changes

## Project Structure

### Documentation (this feature)
```
specs/053-update-to-iris/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   └── test_import_iris_vector_graph.py
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_rag/
├── pipelines/
│   └── hybrid_graphrag_discovery.py  # PRIMARY UPDATE: lines 127-130, 166-169
pyproject.toml                         # DEPENDENCY UPDATE: line 76, 79
tests/
├── contract/
│   └── test_import_requirements.py    # UPDATE: import validation test
└── unit/
    └── test_hybrid_graphrag.py        # UPDATE: remove old import references
```

**Structure Decision**: Single Python package with import path updates in one primary file (`hybrid_graphrag_discovery.py`), dependency constraint update in `pyproject.toml`, and corresponding test updates.

## Phase 0: Outline & Research

**No research needed** - all information is available:

1. **iris-vector-graph 1.1.1 structure** (from user diagnosis):
   - ✅ Top-level module: `iris_vector_graph`
   - ✅ Direct exports: `IRISGraphEngine`, `HybridSearchFusion`, `TextSearchEngine`, `VectorOptimizer`
   - ✅ No `iris_vector_graph_core` submodule

2. **Current import structure** (from code inspection):
   - ❌ Old: `from iris_vector_graph_core.engine import IRISGraphEngine`
   - ✅ New: `from iris_vector_graph import IRISGraphEngine`

3. **Impact analysis** (from grep results):
   - Primary file: `iris_rag/pipelines/hybrid_graphrag_discovery.py`
   - Test files: `tests/unit/test_hybrid_graphrag.py`, `tests/contract/test_import_requirements.py`
   - Spec file: `specs/053-update-to-iris/spec.md` (reference only)

**Output**: research.md documenting the import migration and version constraint change

## Phase 1: Design & Contracts

1. **Data Model** (data-model.md):
   - **Entity**: ImportPaths
   - **Fields**: old_path (iris_vector_graph_core.*), new_path (iris_vector_graph.*)
   - **Relationships**: N/A (no data model changes, pure import refactor)
   - **Validation**: Contract tests ensure new imports work

2. **API Contracts** (/contracts/):
   - Contract test: `test_import_iris_vector_graph.py`
     - FR-001: `from iris_vector_graph import IRISGraphEngine` succeeds
     - FR-002: `from iris_vector_graph import HybridSearchFusion` succeeds
     - FR-003: `from iris_vector_graph import TextSearchEngine` succeeds
     - FR-004: `from iris_vector_graph import VectorOptimizer` succeeds
     - FR-006: `from iris_vector_graph_core.*` fails with ImportError
     - FR-009: Version check test for iris-vector-graph >= 1.1.1

3. **Integration Test Scenarios** (from user stories in spec):
   - Install iris-vector-rag → import HybridGraphRAGPipeline → succeeds
   - Query with HybridGraphRAG → uses new iris_vector_graph modules → succeeds
   - Install with iris-vector-graph < 1.1.1 → fails with clear version error

4. **Agent File Update**:
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
   - Add note about iris-vector-graph 1.1.1+ requirement
   - Preserve existing HybridGraphRAG documentation

**Output**: data-model.md, /contracts/test_import_iris_vector_graph.py, quickstart.md, CLAUDE.md update

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. **Contract Test Tasks** [P]:
   - T001: Write contract test for IRISGraphEngine import
   - T002: Write contract test for HybridSearchFusion import
   - T003: Write contract test for TextSearchEngine import
   - T004: Write contract test for VectorOptimizer import
   - T005: Write contract test for iris_vector_graph_core rejection
   - T006: Write contract test for version validation

2. **Implementation Tasks** (TDD order):
   - T007: Update pyproject.toml dependency (iris-vector-graph>=1.1.1)
   - T008: Update hybrid_graphrag_discovery.py lines 127-130 (installed package imports)
   - T009: Update hybrid_graphrag_discovery.py lines 166-169 (local path imports)
   - T010: Remove iris_vector_graph_core path validation (lines 82-86)
   - T011: Update test files to use new import paths
   - T012: Run contract tests to validate imports
   - T013: Run integration tests with HybridGraphRAG pipeline

3. **Validation Tasks**:
   - T014: Test local installation with new iris-vector-graph 1.1.1
   - T015: Verify all existing HybridGraphRAG tests pass
   - T016: Run quickstart.md validation

**Ordering Strategy**:
- Contract tests first (T001-T006) [P] - can run in parallel
- Dependency update (T007) before implementation (T008-T011)
- Implementation tasks (T008-T011) in file order
- Test validation (T012-T013) after implementation
- Final validation (T014-T016) sequential

**Task Dependencies**:
```
T001-T006 [P] → T007 → T008 → T009 → T010 → T011 → T012 → T013 → T014 → T015 → T016
```

**Estimated Output**: 16 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, verify PyPI package works)

## Complexity Tracking
*No constitutional violations - this section intentionally empty*

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
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.7.1 - See `/.specify/memory/constitution.md`*
