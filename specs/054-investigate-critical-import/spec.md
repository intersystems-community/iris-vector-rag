# Feature Specification: Investigate Common Module Import Issue in iris-vector-rag 0.4.1

**Feature Branch**: `054-investigate-critical-import`
**Created**: 2025-11-09
**Status**: Investigation
**Input**: User description: "Investigate critical import issue in iris-vector-rag 0.4.1: The package is trying to import from common.iris_dbapi_connector and common.iris_connection_manager which may not be included in the PyPI distribution."

## Investigation Summary

### Current Findings (2025-11-09)

**Package Structure**: ✅ CORRECT
- `common` directory exists in source repository
- `common/iris_dbapi_connector.py` exists with `get_iris_dbapi_connection()` function (line 151)
- `common/iris_connection_manager.py` exists with `get_iris_connection()` function (line 342)
- pyproject.toml correctly includes `common*` in package configuration

**Distribution Verification**: ✅ PACKAGED CORRECTLY
- Verified iris_vector_rag-0.4.1-py3-none-any.whl contains 24 files in `common/` directory
- Both `common/iris_dbapi_connector.py` and `common/iris_connection_manager.py` are present in wheel
- Package structure is correct for top-level `common` module import

**Import Locations**:
- `iris_vector_rag/core/connection.py:155` - imports from `common.iris_dbapi_connector`
- `iris_vector_rag/core/connection.py:194` - imports from `common.iris_connection_manager`

### User-Reported Issue

HippoRAG2 pipeline user reports `ModuleNotFoundError: No module named 'common.iris_dbapi_connector'` despite package being correctly built and distributed.

### Hypothesis: Package Namespace Conflict

**Likely Cause**: The `common` module as a top-level package may conflict with other packages or system modules named `common`, causing Python's import system to load the wrong `common` module.

---

## User Scenarios & Testing

### Primary User Story
As a HippoRAG2 pipeline developer, I need iris-vector-rag to connect to IRIS database successfully so that entity extraction can proceed without import errors.

### Acceptance Scenarios
1. **Given** iris-vector-rag 0.4.1 is installed in a clean venv, **When** user imports ConnectionManager, **Then** all `common.*` imports should resolve successfully
2. **Given** iris-vector-rag 0.4.1 is installed alongside other packages, **When** user creates a ConnectionManager instance, **Then** the correct `common.iris_dbapi_connector` module should be imported without conflicts
3. **Given** ConnectionManager._create_connection() is called, **When** importing from common modules, **Then** no ModuleNotFoundError should occur

### Edge Cases
- What happens when another package also has a `common` module in the user's environment?
- How does Python's import system resolve `from common.X import Y` when multiple `common` modules exist?
- Does the issue occur only in certain Python versions or with specific setuptools/pip versions?

## Requirements

### Functional Requirements
- **FR-001**: iris-vector-rag MUST successfully import `common.iris_dbapi_connector` when ConnectionManager creates IRIS connections
- **FR-002**: iris-vector-rag MUST successfully import `common.iris_connection_manager` when ConnectionManager creates database connections
- **FR-003**: Package distribution MUST include all `common` module files required for connection functionality
- **FR-004**: iris-vector-rag MUST work correctly when installed via pip in a fresh virtual environment
- **FR-005**: ConnectionManager MUST not fail with ModuleNotFoundError when common module imports are attempted

### Non-Functional Requirements
- **NFR-001**: Import mechanism MUST be robust against namespace conflicts from other installed packages
- **NFR-002**: Package structure MUST follow Python packaging best practices to avoid import shadowing
- **NFR-003**: Solution MUST be backward compatible with existing installations or require BREAKING CHANGE version bump

### Key Entities
- **ConnectionManager**: Component in iris_vector_rag that manages IRIS database connections
- **common module**: Top-level Python package containing IRIS connection utilities
- **iris_dbapi_connector**: Module within common package providing DBAPI connection functionality
- **iris_connection_manager**: Module within common package providing connection management utilities

---

## Investigation Plan

### Phase 1: Reproduce Issue
1. Create clean Python 3.12 venv
2. Install iris-vector-rag from PyPI
3. Attempt to import ConnectionManager and create connection
4. Document exact error message and stack trace
5. Verify if `common` module is actually present in site-packages

### Phase 2: Diagnose Root Cause
1. Check if `common` module is on sys.path
2. Inspect sys.modules for common conflicts
3. Test which common module is loaded
4. Compare working installation vs broken installation
5. Identify any shadowing packages

### Phase 3: Propose Solutions
- **Option A**: Rename common to iris_vector_rag_common (requires version bump)
- **Option B**: Move common inside iris_vector_rag package (BREAKING CHANGE)
- **Option C**: Document installation requirements and potential conflicts

### Phase 4: Implement Fix
1. Choose solution based on diagnosis
2. Update imports in affected files
3. Rebuild package
4. Test in clean environment
5. Verify fix resolves HippoRAG2 pipeline issue

### Phase 5: Release
1. Update version appropriately
2. Update CHANGELOG with fix details
3. Build and validate package
4. Publish to PyPI
5. Notify affected users

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Initial investigation completed
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Investigation plan created
- [ ] Issue reproduced in clean environment
- [ ] Root cause diagnosed
- [ ] Solution selected
- [ ] Fix implemented
- [ ] Fix tested and validated
- [ ] Release completed
