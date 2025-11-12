# Implementation Plan: Investigate Common Module Import Issue

**Branch**: `054-investigate-critical-import` | **Date**: 2025-11-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/054-investigate-critical-import/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✓
2. Fill Technical Context ✓
3. Fill Constitution Check ✓
4. Execute Phase 0 → research.md (reproduce issue, diagnose root cause)
5. Execute Phase 1 → contracts (solution validation tests), quickstart.md (fix verification)
6. Re-evaluate Constitution Check
7. Plan Phase 2 → Task generation approach
8. STOP - Ready for /tasks command
```

## Summary

**Problem**: HippoRAG2 pipeline user reports `ModuleNotFoundError: No module named 'common.iris_dbapi_connector'` when using iris-vector-rag 0.4.1, despite package verification showing the module IS correctly packaged and distributed.

**Hypothesis**: Namespace conflict - Python's import system is loading a different `common` package instead of iris-vector-rag's top-level `common` module.

**Technical Approach**:
1. Reproduce issue in clean venv with iris-vector-rag from PyPI
2. Diagnose whether namespace shadowing is occurring
3. Implement fix (most likely: move common inside iris_vector_rag package)
4. Test fix in HippoRAG2 pipeline environment
5. Release patched version to PyPI

## Technical Context

**Language/Version**: Python 3.10+ (per iris-vector-rag requirements)
**Primary Dependencies**:
- intersystems-irispython>=5.1.2 (IRIS database connectivity)
- iris-vector-rag 0.4.1 (current broken version)

**Storage**: InterSystems IRIS vector database
**Testing**: pytest (existing test infrastructure)
**Target Platform**: Cross-platform (Linux, macOS, Windows)
**Project Type**: Single project (Python package)

**Performance Goals**: N/A (bug fix)
**Constraints**:
- BREAKING CHANGE acceptable (already communicated to users)
- Must maintain backward compatibility with IRIS connection interfaces
- Fix must work in both development and pip-installed environments

**Scale/Scope**: Affects 2 files (iris_vector_rag/core/connection.py), 2 modules (common.iris_dbapi_connector, common.iris_connection_manager)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ✓ Bug fix maintains RAGPipeline architecture | ✓ No application logic changes | ✓ CLI unchanged

**II. Pipeline Validation & Requirements**: ✓ Will add import validation test | ✓ Setup remains idempotent

**III. Test-Driven Development**: ✓ Contract test will verify import works | ✓ N/A (not performance-related)

**IV. Performance & Enterprise Scale**: ✓ No performance impact | ✓ IRIS operations unchanged

**V. Production Readiness**: ✓ Logging unchanged | ✓ Health checks unchanged | ✓ Docker deployment unaffected

**VI. Explicit Error Handling**: ✓ Will fail fast if modules missing | ✓ ImportError provides clear message | ✓ Points to installation issue

**VII. Standardized Database Interfaces**: ✓ Uses existing iris_dbapi_connector | ✓ No ad-hoc queries | ✓ Patterns unchanged

**Result**: ✅ PASS - This is a packaging bug fix that maintains all constitutional principles

## Project Structure

### Documentation (this feature)
```
specs/054-investigate-critical-import/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (reproduce + diagnose)
├── quickstart.md        # Phase 1 output (fix verification steps)
└── contracts/           # Phase 1 output (import validation tests)
    └── test_common_imports.py
```

### Source Code (repository root)
```
iris_vector_rag/
├── core/
│   └── connection.py           # File with problematic imports (lines 155, 194)
├── common/                      # NEW: Move common inside package
│   ├── __init__.py
│   ├── iris_dbapi_connector.py
│   └── iris_connection_manager.py

common/                          # OLD: Top-level module (to be removed)
├── iris_dbapi_connector.py
└── iris_connection_manager.py

tests/
├── contract/
│   └── test_common_imports.py   # NEW: Verify imports work after fix
└── integration/
    └── test_connection_manager.py  # Update to use new import path
```

**Structure Decision**: Move `common` directory inside `iris_vector_rag` package to avoid namespace conflicts. This prevents Python from loading other `common` packages instead of ours.

**Alternative Considered**: Rename to `iris_vector_rag_common` - rejected because moving inside package is cleaner and more Pythonic.

## Phase 0: Outline & Research

### Research Tasks

1. **Reproduce the issue**:
   - Create clean Python 3.12 venv
   - `pip install iris-vector-rag==0.4.1`
   - Try to import ConnectionManager and trigger common module imports
   - Document exact error and traceback

2. **Diagnose namespace conflict**:
   - Check `sys.modules` for 'common' before/after import attempt
   - Use `python -v` to trace import resolution
   - Identify which package is providing the conflicting `common` module
   - Verify if issue occurs in isolated environment vs HippoRAG2 environment

3. **Research Python import best practices**:
   - Top-level namespace pollution risks
   - Package-relative imports vs absolute imports
   - Common patterns for utility modules (inside vs outside package)
   - Migration strategies for breaking import changes

4. **Evaluate fix options**:
   - **Option A**: Move common → iris_vector_rag/common (RECOMMENDED)
     - Pros: Avoids namespace conflicts, Pythonic, clear ownership
     - Cons: BREAKING CHANGE for any external code importing common directly

   - **Option B**: Rename to iris_vector_rag_common (top-level)
     - Pros: No structural change
     - Cons: Still pollutes top-level namespace, less clean

   - **Option C**: Keep as-is, document conflicts
     - Pros: No breaking change
     - Cons: Doesn't fix the issue

**Output**: research.md with reproduction steps, diagnosis findings, and selected fix approach

## Phase 1: Design & Contracts

### 1. Data Model

**Entity**: CommonModuleLocation
- `old_path`: str = "common.iris_dbapi_connector"
- `new_path`: str = "iris_vector_rag.common.iris_dbapi_connector"
- `affected_files`: List[str] = ["iris_vector_rag/core/connection.py"]

**Migration**: One-time import path update

### 2. API Contracts

**Contract 1**: Import Resolution
```python
# File: contracts/test_common_imports.py

def test_import_iris_dbapi_connector():
    """Verify iris_dbapi_connector can be imported without namespace conflict."""
    from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection
    assert callable(get_iris_dbapi_connection)

def test_import_iris_connection_manager():
    """Verify iris_connection_manager can be imported without namespace conflict."""
    from iris_vector_rag.common.iris_connection_manager import get_iris_connection
    assert callable(get_iris_connection)

def test_connection_manager_imports():
    """Verify ConnectionManager can import its dependencies."""
    from iris_vector_rag.core.connection import ConnectionManager
    # Should not raise ImportError
    assert ConnectionManager is not None
```

### 3. Migration Contract

**Contract 2**: Backward Compatibility (if needed)
```python
# File: contracts/test_backward_compat.py

def test_old_common_imports_show_clear_error():
    """Verify old import paths show clear migration message."""
    with pytest.raises(ImportError, match="moved to iris_vector_rag.common"):
        from common.iris_dbapi_connector import get_iris_dbapi_connection
```

### 4. Quickstart Verification

**File**: quickstart.md
```markdown
# Fix Verification Steps

## Prerequisites
- Python 3.10+
- Clean virtual environment

## Steps

1. Install patched version:
   ```bash
   pip install iris-vector-rag==0.4.2  # or 0.5.0 if BREAKING
   ```

2. Verify imports work:
   ```python
   from iris_vector_rag.core.connection import ConnectionManager
   # Should not raise ImportError
   ```

3. Test in HippoRAG2 environment:
   ```bash
   cd ../hipporag2-pipeline
   pip install --upgrade iris-vector-rag==0.4.2
   python tests/test_e2e_simple.py  # Should complete without hanging
   ```

## Success Criteria
- No ImportError when importing ConnectionManager
- HippoRAG2 entity extraction logging appears
- No "hanging" during indexing
```

### 5. Update CLAUDE.md

Run incremental agent context update:
```bash
.specify/scripts/bash/update-agent-context.sh claude
```

Add to recent changes:
- Common module moved inside iris_vector_rag package (v0.4.2/0.5.0)
- Import paths updated: `common.X` → `iris_vector_rag.common.X`
- Namespace conflict resolved

**Output**: contracts/test_common_imports.py, quickstart.md, updated CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Load `.specify/templates/tasks-template.md` as base
2. Generate tasks from Phase 0 research (reproduction, diagnosis)
3. Generate tasks from Phase 1 contracts (import validation tests)
4. Generate implementation tasks (file moves, import updates)
5. Generate testing tasks (contract tests, HippoRAG2 validation)
6. Generate release tasks (version bump, PyPI publish)

**Task Ordering**:
1. **Research & Validation** [P]:
   - Reproduce issue in clean environment
   - Diagnose namespace conflict
   - Write contract tests (TDD)

2. **Implementation** (sequential):
   - Move common/ → iris_vector_rag/common/
   - Update imports in iris_vector_rag/core/connection.py
   - Update pyproject.toml if needed
   - Update any other internal imports

3. **Testing** [P]:
   - Run contract tests (should pass)
   - Run existing test suite (verify no regressions)
   - Test in HippoRAG2 environment

4. **Release** (sequential):
   - Decide version: 0.4.2 (patch) vs 0.5.0 (BREAKING if external common imports exist)
   - Update __version__
   - Build package
   - Verify wheel contents
   - Publish to PyPI
   - Update HippoRAG2 to use new version

**Estimated Output**: 12-15 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (move files, update imports, test)
**Phase 5**: Validation (contract tests pass, HippoRAG2 works, no regressions)

## Complexity Tracking

No constitutional violations - this is a straightforward packaging bug fix.

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (research.md)
- [x] Phase 1: Design complete (contracts/test_common_imports.py, quickstart.md)
- [ ] Phase 2: Task planning complete (awaiting /tasks command)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS (no design changes to constitutional components)
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)
- [x] Contract tests written (7 tests, should initially FAIL)
- [x] Quickstart verification procedure documented

**Next Command**: `/tasks` to generate detailed task list

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
