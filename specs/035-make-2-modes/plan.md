# Implementation Plan: Configurable Test Backend Modes (Enterprise & Community)

**Branch**: `035-make-2-modes` | **Date**: 2025-10-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/Users/tdyar/ws/rag-templates/specs/035-make-2-modes/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✓
2. Fill Technical Context ✓
3. Fill Constitution Check section ✓
4. Evaluate Constitution Check → In Progress
5. Execute Phase 0 → research.md
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
7. Re-evaluate Constitution Check
8. Plan Phase 2 → tasks.md generation approach
9. STOP - Ready for /tasks command
```

## Summary

Add configurable test backend modes (enterprise and community) to handle different IRIS database license constraints during testing. Community mode limits connections to 1 concurrent to prevent license exhaustion, while enterprise mode allows unlimited parallel execution. Integration with iris-devtools (required dev dependency from ../iris-devtools) provides container lifecycle management, schema reset, connection validation, and health checks. Configuration via config file with environment variable override.

**Primary Goal**: Enable reliable test execution across both IRIS Community and Enterprise editions by automatically applying appropriate connection limits and execution strategies based on detected database edition.

## Technical Context

**Language/Version**: Python 3.12 (existing project standard)
**Primary Dependencies**:
- pytest (existing test framework)
- iris-devtools from ../iris-devtools (required dev dependency - NEW)
- intersystemsdc/iris-community:latest (for community testing)
- docker.iscinternal.com/intersystems/iris:2025.3.0EHAT.127.0-linux-arm64v8 (for enterprise testing)

**Storage**: InterSystems IRIS database (existing)
**Testing**: pytest with database-dependent markers (@pytest.mark.requires_database)
**Target Platform**: Development environments (macOS, Linux) with Docker support
**Project Type**: Single Python project (RAG framework)

**Performance Goals**:
- Community mode: Prevent >95% of license exhaustion errors
- Enterprise mode: No performance degradation from safeguards
- Test execution: <5 minutes for full suite per backend mode

**Constraints**:
- Community mode: Maximum 1 concurrent database connection
- Enterprise mode: Unlimited concurrent connections
- iris-devtools must be available at ../iris-devtools (sibling directory)
- Backend mode validated once at test session start (no mid-session switching)
- Edition mismatch (config vs actual IRIS) must fail fast with clear error

**Scale/Scope**:
- 2 backend modes (community, enterprise)
- 4 core operations via iris-devtools (container lifecycle, schema reset, connection validation, health checks)
- ~25-30 new tests validating mode selection and behavior
- Configuration: 1 config file setting + 1 environment variable override

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Configuration component designed as reusable framework element
- ✓ No application-specific logic (works for any pytest-based IRIS project)
- ✓ CLI interface via pytest options and Make targets

**II. Pipeline Validation & Requirements**:
- ✓ Automated validation: iris-devtools presence check
- ✓ Automated validation: IRIS edition detection vs configured mode
- ✓ Setup procedures idempotent (mode detection runs identically each time)
- ✓ Fail-fast with clear errors for missing dependencies or mismatches

**III. Test-Driven Development**:
- ✓ Contract tests for backend mode configuration required
- ✓ Contract tests for iris-devtools integration required
- ✓ Integration tests for mode switching and validation required
- ✓ Tests against live IRIS database (both community and enterprise editions)
- ⚠️  Performance tests: Validate >95% license error prevention in community mode

**IV. Performance & Enterprise Scale**:
- ✓ Enterprise mode supports parallel execution without limits
- ✓ Community mode uses sequential execution to prevent connection pooling issues
- ✓ Connection pooling optimized per mode (1 for community, unlimited for enterprise)
- N/A Incremental indexing (not applicable to test infrastructure)

**V. Production Readiness**:
- ✓ Structured logging: Log backend mode at test session start
- ✓ Health checks: iris-devtools health/readiness validation
- ⚠️  Docker deployment: Needs documentation for both community/enterprise test images
- ✓ Configuration externalized: Config file + env var override

**VI. Explicit Error Handling**:
- ✓ No silent failures: All validation errors must fail fast
- ✓ Clear exception messages required for:
  * Missing iris-devtools dependency
  * IRIS edition mismatch
  * Connection limit exceeded in community mode
- ✓ Actionable error context: Messages must tell user how to fix (e.g., "Set IRIS_BACKEND_MODE=community or update config")

**VII. Standardized Database Interfaces**:
- ✓ Use existing common/iris_connection_manager.py for connections
- ✓ Use existing common/iris_port_discovery.py for IRIS instance detection
- ✓ iris-devtools provides standardized container/schema management
- ✓ No ad-hoc IRIS queries (all database state management via iris-devtools)

**Constitution Violations**: None identified

**Complexity Justifications**: N/A

## Project Structure

### Documentation (this feature)
```
specs/035-make-2-modes/
├── plan.md              # This file (/plan command output)
├── spec.md              # Feature specification (completed)
├── research.md          # Phase 0 output (to be created)
├── data-model.md        # Phase 1 output (to be created)
├── quickstart.md        # Phase 1 output (to be created)
├── contracts/           # Phase 1 output (to be created)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Single Python project structure (existing)
iris_rag/                        # Main framework package
├── config/
│   ├── backend_modes.py         # NEW: Backend mode configuration
│   └── default_config.yaml      # MODIFIED: Add backend_mode setting
├── testing/                     # NEW: Testing infrastructure
│   ├── backend_manager.py       # NEW: Backend mode manager
│   ├── iris_devtools_bridge.py  # NEW: iris-devtools integration
│   └── validators.py            # NEW: Edition detection & validation
└── utils/                       # Existing utilities
    └── connection_pool.py       # MODIFIED: Add mode-aware pooling

tests/
├── conftest.py                  # MODIFIED: Add backend mode fixtures
├── contract/                    # Existing contract tests
│   ├── test_backend_mode_config.py  # NEW: Backend config contracts
│   ├── test_edition_detection.py    # NEW: Edition detection contracts
│   └── test_iris_devtools_integration.py  # NEW: iris-devtools contracts
├── integration/                 # Existing integration tests
│   ├── test_community_mode_execution.py  # NEW: Community mode validation
│   └── test_enterprise_mode_execution.py # NEW: Enterprise mode validation
└── unit/                        # Existing unit tests
    └── test_backend_validators.py  # NEW: Validator unit tests

common/                          # Existing shared utilities (unchanged)
├── iris_connection_manager.py   # MODIFIED: Add mode-aware connection logic
└── iris_port_discovery.py       # Unchanged

.specify/                        # Existing spec infrastructure
└── config/
    └── backend_modes.yaml       # NEW: Backend mode configuration schema
```

**Structure Decision**: Single Python project structure maintained. New testing infrastructure components added under `iris_rag/testing/` to keep framework-first architecture. Configuration extends existing `iris_rag/config/` patterns. Tests follow existing pytest structure with contract, integration, and unit divisions.

## Phase 0: Outline & Research
*Status: Starting*

### Unknowns to Research

1. **iris-devtools API and Integration Patterns**
   - Research task: Document iris-devtools API for container lifecycle, schema reset, connection validation, health checks
   - Rationale: Need to understand how to properly integrate iris-devtools as a sibling dependency
   - Output: API contracts for iris-devtools bridge component

2. **IRIS Edition Detection Methods**
   - Research task: Find reliable method to detect IRIS Community vs Enterprise edition at runtime
   - Rationale: Must distinguish editions to validate backend mode configuration
   - Alternatives: SQL query, connection properties, license file detection
   - Output: Edition detection implementation strategy

3. **pytest Connection Pooling Behavior**
   - Research task: Understand how pytest manages database connections across test sessions
   - Rationale: Need to enforce 1-connection limit for community mode without breaking existing tests
   - Output: Connection pool integration approach

4. **Configuration Override Precedence**
   - Research task: Establish clear precedence for config file vs environment variable
   - Rationale: Users need predictable behavior when both are set
   - Output: Configuration resolution order specification

**Next Step**: Generate research.md with consolidated findings

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

### Data Model (`data-model.md`)

**Entities**:

1. **BackendMode** (Enum)
   - Values: COMMUNITY, ENTERPRISE
   - Validation: Must be one of two values
   - Source: Configuration file or environment variable

2. **BackendConfiguration**
   - Fields:
     * mode: BackendMode (required)
     * max_connections: int (1 for COMMUNITY, unlimited for ENTERPRISE)
     * execution_strategy: ExecutionStrategy (SEQUENTIAL or PARALLEL)
     * iris_devtools_path: Path (default: ../iris-devtools)
   - Validation: mode must match detected IRIS edition
   - State: Immutable after test session start

3. **IRISEdition** (Enum)
   - Values: COMMUNITY, ENTERPRISE
   - Source: Runtime detection from IRIS connection
   - Validation: Must be detectable before test execution

4. **DatabaseStateManager**
   - Fields:
     * iris_devtools: IrisDevToolsBridge
     * backend_config: BackendConfiguration
   - Operations:
     * start_container()
     * stop_container()
     * reset_schema()
     * validate_connection()
     * check_health()
   - Behavior varies by backend_config.mode

### API Contracts (`/contracts/`)

Will generate OpenAPI-style contracts for:

1. **Configuration API** (`backend_config_contract.yaml`)
   - GET /config/backend-mode → Returns current mode
   - POST /config/validate → Validates configuration
   - Errors: IRIS_DEVTOOLS_MISSING, EDITION_MISMATCH

2. **iris-devtools Bridge API** (`iris_devtools_contract.yaml`)
   - POST /container/start → Start IRIS container
   - POST /container/stop → Stop IRIS container
   - POST /schema/reset → Reset database schema
   - GET /connection/validate → Validate connection
   - GET /health → Check IRIS health
   - Errors: DEVTOOLS_UNAVAILABLE, CONTAINER_FAILED

3. **Edition Detection API** (`edition_detection_contract.yaml`)
   - GET /iris/edition → Detect IRIS edition
   - Returns: COMMUNITY or ENTERPRISE
   - Errors: DETECTION_FAILED, CONNECTION_FAILED

### Contract Tests

Generated from contracts above:

1. `test_backend_mode_config.py` - FR-001, FR-002, FR-009, FR-012
2. `test_edition_detection.py` - FR-008
3. `test_iris_devtools_integration.py` - FR-006, FR-007, FR-013
4. `test_connection_pooling.py` - FR-003, FR-011
5. `test_execution_strategies.py` - FR-004, FR-005

All tests must fail initially (no implementation).

### Quickstart Guide (`quickstart.md`)

Structure:
1. Prerequisites (iris-devtools at ../iris-devtools, Docker running)
2. Configuration (create config file with backend_mode setting)
3. Running tests in community mode (example command)
4. Running tests in enterprise mode (example command)
5. Environment variable override (example)
6. Troubleshooting (common errors and fixes)

### Agent Context Update

Execute: `.specify/scripts/bash/update-agent-context.sh claude`
- Add: iris-devtools integration patterns
- Add: Backend mode configuration approach
- Add: IRIS edition detection method
- Keep recent: Feature 034 (Testing Gaps), Feature 035 (Backend Modes)
- Maintain <150 lines

**Outputs**: data-model.md, contracts/*.yaml, 5+ contract test files, quickstart.md, CLAUDE.md updated

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate from Phase 1 artifacts:
  * Each contract → contract test task [P]
  * Each data model entity → model implementation task [P]
  * Each iris-devtools operation → bridge method task
  * Each validation rule → validator implementation task
  * Integration tests for mode switching
  * Documentation tasks for quickstart examples

**Ordering Strategy**:
1. **Setup Phase** (Sequential):
   - T001: Research iris-devtools API → research.md
   - T002: Research IRIS edition detection → research.md
   - T003: Design data model → data-model.md
   - T004: Generate API contracts → contracts/

2. **Contract Test Phase** (Parallel after T004):
   - T005-T009: Write contract tests [P]

3. **Implementation Phase** (TDD order):
   - T010: Implement BackendMode enum
   - T011: Implement IRISEdition enum
   - T012: Implement edition detection
   - T013: Implement BackendConfiguration
   - T014: Implement iris-devtools bridge [P]
   - T015: Implement connection pool manager [P]
   - T016: Implement pytest fixtures
   - T017: Update conftest.py

4. **Integration Phase**:
   - T018: Community mode integration test
   - T019: Enterprise mode integration test
   - T020: Mode switching integration test

5. **Documentation Phase** [P]:
   - T021: Write quickstart.md
   - T022: Update CLAUDE.md
   - T023: Add Make targets for mode selection

**Estimated Output**: 25-30 tasks in dependency order with [P] markers

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (via /tasks or /implement commands)
**Phase 4**: Validation (run contract tests, verify all pass)
**Phase 5**: Documentation finalization

## Progress Tracking

- [x] Specification loaded and analyzed
- [x] Technical context filled
- [x] Constitution check completed (no violations)
- [x] Phase 0: research.md generated
- [x] Phase 1: data-model.md generated
- [x] Phase 1: contracts/ generated
- [⚠️] Phase 1: Contract tests generated (deferred to /tasks command per template)
- [x] Phase 1: quickstart.md generated
- [x] Phase 1: Agent context updated
- [x] Constitution re-check passed (no new violations)
- [x] Phase 2 approach documented
- [x] Ready for /tasks command

## Artifacts Generated

**Phase 0 (Research)**:
- ✅ `/Users/tdyar/ws/rag-templates/specs/035-make-2-modes/research.md`
  * iris-devtools API integration patterns
  * IRIS edition detection via SQL query
  * pytest connection pooling with Semaphore
  * Configuration precedence: env var > config > default

**Phase 1 (Design)**:
- ✅ `/Users/tdyar/ws/rag-templates/specs/035-make-2-modes/data-model.md`
  * 5 entity definitions with validation rules
  * State transition diagrams
  * Error hierarchy
  * Relationship diagrams
- ✅ `/Users/tdyar/ws/rag-templates/specs/035-make-2-modes/contracts/`
  * backend_config_contract.yaml (8 test scenarios)
  * README.md (contract format specification)
- ✅ `/Users/tdyar/ws/rag-templates/specs/035-make-2-modes/quickstart.md`
  * Prerequisites and setup instructions
  * Configuration examples (file + env var)
  * Running tests in both modes
  * Troubleshooting guide
- ✅ `/Users/tdyar/ws/rag-templates/CLAUDE.md`
  * Updated with Python 3.12 context
  * Added IRIS database references

**Note**: Contract test generation deferred to /tasks command as contract tests are implementation tasks, not design artifacts.

## ✅ Tasks Generated

**Generated**: `/Users/tdyar/ws/rag-templates/specs/035-make-2-modes/tasks.md`

**Task Summary**:
- **30 tasks total** in TDD order
- **Phase 3.1**: Setup & Prerequisites (T001-T004)
- **Phase 3.2**: Tests First - TDD (T005-T015) - 11 contract & integration tests [MUST FAIL]
- **Phase 3.3**: Core Implementation (T016-T023) - 8 implementation tasks [MAKE TESTS PASS]
- **Phase 3.4**: Integration & Validation (T024-T029) - 6 integration & validation tasks
- **Phase 3.5**: Polish & Documentation (T030) - 1 documentation task

**Parallel Execution Opportunities**:
- T005-T015: All 11 tests can run in parallel (different files)
- T016, T017, T018, T020, T026: 5 implementation tasks (different files)

**Critical Dependencies**:
- ALL tests (T005-T015) MUST be written and MUST FAIL before implementation (T016-T023)
- T027 (validate contract tests pass) depends on T016-T026
- T028-T029 (integration validation) depend on T027

## Next Steps

**Immediate**: Review tasks.md, then run `/implement` to execute

**Implementation Workflow**:
1. Execute T001-T004 (setup)
2. Execute T005-T015 in parallel (write failing tests)
3. Verify all tests fail (TDD requirement)
4. Execute T016-T023 sequentially (make tests pass)
5. Execute T024-T029 (integration & validation)
6. Execute T030 (documentation)

**Success Criteria**:
- All 30 tasks completed ✓
- All contract tests pass ✓
- Community mode: >95% license error prevention ✓
- Enterprise mode: Parallel execution works ✓
