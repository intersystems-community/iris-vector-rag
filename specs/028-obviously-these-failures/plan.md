# Implementation Plan: Test Infrastructure Resilience and Database Schema Management

**Branch**: `028-obviously-these-failures` | **Date**: 2025-10-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/intersystems-community/ws/rag-templates/specs/028-obviously-these-failures/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✓ Loaded spec.md successfully
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → ✓ No NEEDS CLARIFICATION markers (infrastructure spec)
   → Project Type: Test infrastructure (pytest framework)
   → Structure Decision: Single project with test utilities
3. Fill the Constitution Check section
   → ✓ Completed constitution alignment analysis
4. Evaluate Constitution Check section
   → ✓ No violations - infrastructure feature supporting TDD
   → Update Progress Tracking: Initial Constitution Check PASS
5. Execute Phase 0 → research.md
   → ✓ No unknowns to research (clear requirements)
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   → ✓ Generated all Phase 1 artifacts
7. Re-evaluate Constitution Check section
   → ✓ No new violations after design
   → Update Progress Tracking: Post-Design Constitution Check PASS
8. Plan Phase 2 → Task generation approach documented
9. STOP - Ready for /tasks command
   → ✓ Plan complete
```

## Summary

This feature implements automated database schema management and test infrastructure resilience to resolve 287 test failures (69 schema errors + 218 cascading failures). The system will automatically detect schema mismatches, provide clean database state for each test class, properly handle contract tests for unimplemented features (MCP), and ensure medical-grade test reliability.

**Primary Requirement**: Eliminate test infrastructure failures by implementing automatic schema validation, reset capabilities, test isolation with cleanup handlers, and contract test markers for TDD compliance.

**Technical Approach**: Extend pytest's conftest.py with custom fixtures for schema management, implement SchemaValidator utility for detecting table mismatches, create TestFixtureManager for database isolation, and add pytest plugin for contract test handling. All operations will be idempotent, logged for audit trails, and optimized for minimal overhead (<100ms per test class).

## Technical Context

**Language/Version**: Python 3.12 (existing test framework)
**Primary Dependencies**: pytest 8.4.1, pytest-cov, iris.dbapi, python-dotenv
**Storage**: InterSystems IRIS database (existing: iris_db_rag_templates container on ports 11972/15273)
**Testing**: pytest with custom fixtures and plugins (conftest.py, plugins/)
**Target Platform**: macOS/Linux development environments + CI/CD runners
**Project Type**: Single project - Test infrastructure utilities
**Performance Goals**:
- Schema reset: <5 seconds for 4 tables
- Test isolation overhead: <100ms per test class
- Pre-flight checks: <2 seconds total
**Constraints**:
- Must not affect production databases (test-only)
- Must be idempotent (safe to run multiple times)
- Must support concurrent test execution
- Must handle partial/dirty database state
**Scale/Scope**: 771 tests across contract/integration/e2e/unit categories

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component is test infrastructure utility (framework support)
- ✓ No application-specific logic (pure test infrastructure)
- ✓ CLI interface via pytest commands and Make targets

**II. Pipeline Validation & Requirements**:
- ✓ Automated requirement validation (pre-flight checks)
- ✓ Setup procedures idempotent (schema reset, cleanup)

**III. Test-Driven Development**:
- ✓ Contract tests exist (Feature 026 quality framework - 100% passing)
- ✓ This feature ENABLES TDD by fixing test infrastructure
- ✓ Performance validation included (schema reset <5s, overhead <100ms)

**IV. Performance & Enterprise Scale**:
- N/A - Infrastructure feature (but optimized: <5s schema reset, <100ms overhead)

**V. Production Readiness**:
- ✓ Structured logging (audit trail for schema operations with timestamps)
- ✓ Health checks (pre-flight IRIS connectivity validation)
- ✓ Docker deployment ready (works with existing docker-compose.yml)

**VI. Explicit Error Handling**:
- ✓ No silent failures (all errors raised with context)
- ✓ Clear exception messages (includes SQLCODE and remediation)
- ✓ Actionable error context (table name, SQL attempted, error details)

**VII. Standardized Database Interfaces**:
- ✓ Uses existing iris_connector.py and iris_connection_manager.py
- ✓ No ad-hoc IRIS queries (uses framework's common/iris_sql_utils.py)
- ✓ New schema validation patterns will be added to common/database_schema_manager.py

**Constitutional Compliance**: PASS ✓
- All principles satisfied
- Supports TDD (Principle III) by fixing test infrastructure
- Enables medical-grade quality validation
- No violations or exceptions required

## Project Structure

### Documentation (this feature)
```
specs/028-obviously-these-failures/
├── plan.md              # This file (/plan command output) ✓
├── spec.md              # Feature specification ✓
├── research.md          # Phase 0 output (minimal - no unknowns) ✓
├── data-model.md        # Phase 1 output ✓
├── quickstart.md        # Phase 1 output ✓
├── contracts/           # Phase 1 output ✓
│   ├── schema_manager_contract.py
│   ├── test_fixtures_contract.py
│   └── contract_tests_contract.py
└── tasks.md             # Phase 2 output (/tasks command - NOT created yet)
```

### Source Code (repository root)
```
# Test infrastructure utilities
tests/
├── conftest.py                          # MODIFIED: Add schema management fixtures
├── plugins/                             # MODIFIED: Add contract test plugin
│   ├── coverage_warnings.py            # Existing (Feature 026)
│   ├── error_message_validator.py      # Existing (Feature 026)
│   ├── tdd_compliance.py               # Existing (Feature 026)
│   └── contract_test_marker.py         # NEW: Handle contract test failures
├── fixtures/                            # NEW: Test fixture helpers
│   ├── __init__.py
│   ├── database_cleanup.py             # Database state cleanup utilities
│   └── schema_reset.py                 # Schema reset utilities
└── utils/                               # NEW: Test utilities
    ├── __init__.py
    ├── schema_validator.py              # Schema validation logic
    └── preflight_checks.py              # Pre-test validation

# Core framework utilities (reusable)
common/
├── iris_connector.py                    # Existing - used for connections
├── iris_connection_manager.py           # Existing - used for connection pooling
├── iris_sql_utils.py                    # Existing - used for SQL helpers
└── database_schema_manager.py           # MODIFIED: Add schema validation methods

# Test markers and configuration
pytest.ini                               # MODIFIED: Add contract marker configuration
```

**Structure Decision**: Single project structure with test infrastructure utilities in `tests/` directory. Reusable framework components added to `common/` for potential use in other projects. Follows existing rag-templates project organization.

## Phase 0: Outline & Research

**Research Status**: No unknowns to research - this is a well-defined infrastructure feature with clear requirements.

### Technical Context Analysis
All technical details are known:
- ✓ Python 3.12 with pytest 8.4.1 (existing)
- ✓ IRIS database connection via iris.dbapi (existing)
- ✓ Schema structure known (SourceDocuments, DocumentChunks, Entities, Relationships)
- ✓ Test categories defined (contract, integration, e2e, unit)
- ✓ Performance targets specified in NFRs

**Output**: See research.md for consolidated findings ✓

## Phase 1: Design & Contracts

### 1. Data Model
See data-model.md ✓

### 2. API Contracts
See contracts/ directory ✓

### 3. Test Scenarios
See quickstart.md ✓

### 4. Agent Context Update
Updated via `.specify/scripts/bash/update-agent-context.sh claude` ✓

**Phase 1 Complete**: All artifacts generated ✓

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Load `.specify/templates/tasks-template.md` as base structure
2. Extract tasks from Phase 1 contracts and data model
3. Generate TDD-ordered task list

**Estimated Output**: 18-22 numbered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md, then implementation begins)

**Phase 4**: Validation (verify 0 schema errors, contract tests properly marked)

## Progress Tracking

### Phases Completed
- [x] Phase 0: Research complete
- [x] Phase 1: Design and contracts complete
- [ ] Phase 2: Task generation (awaiting /tasks command)
- [ ] Phase 3: Implementation (awaiting execution)
- [ ] Phase 4: Validation (final verification)

### Constitution Checks
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS

### Gate Status
- [x] Ready for /tasks command ✓

## Next Steps

**Ready for /tasks command** ✓

Expected outcome: 18-22 tasks covering schema validation, test fixtures, cleanup handlers, pytest plugins, and integration.
