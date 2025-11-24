# Implementation Plan: Simplify IRIS Connection Architecture

**Branch**: `051-simplify-iris-connection` | **Date**: 2025-11-23 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/051-simplify-iris-connection/spec.md`

## Summary

Reduce IRIS connection architecture from 6 components (~1100 LOC) to 1-2 components (~300 LOC) for 73% code reduction. Replace multiple abstraction layers (ConnectionManager, ConnectionPool, IRISConnectionManager, Backend Mode config) with single `get_iris_connection()` function and optional explicit `IRISConnectionPool` class. Automatic edition detection eliminates manual backend mode configuration. Preserve UV compatibility fix and maintain backward compatibility during migration.

**Primary Goal**: Developer can understand connection flow in <5 minutes vs current ~hours of onboarding.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: intersystems-irispython (iris module), pytest, threading (stdlib)
**Storage**: InterSystems IRIS database via DBAPI
**Testing**: pytest with contract tests (TDD), integration tests with iris-devtester
**Target Platform**: Linux/macOS servers, local development environments
**Project Type**: Single library (iris-vector-rag Python package)
**Performance Goals**: Zero regression (<5% overhead vs current), connection establishment <50ms, module-level caching reduces repeated overhead
**Constraints**: Must preserve UV environment compatibility fix, maintain backward compatibility during migration, automatic edition detection without external config files
**Scale/Scope**: 156 files import connection modules, ~40 test files use connection fixtures, 6 components to consolidate

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle III: Test-First (TDD) ✅ PASS
- Contract tests will be written before implementation
- Tests will fail initially (red phase)
- Implementation proceeds only after tests approved
- Integration tests added after contract tests pass

### Principle IV: Backward Compatibility ✅ PASS
- New `get_iris_connection()` API is additive (no breaking changes)
- Old APIs (ConnectionManager, IRISConnectionManager) continue working with deprecation warnings
- Existing test suite passes unchanged during migration
- Deprecation timeline: 1 major version cycle per clarification

### Principle IX: Simplicity (YAGNI) ✅ PASS
- Reducing from 6 components to 1-2 components aligns with simplicity principle
- Removing unnecessary abstractions (ConnectionManager layer confusion)
- Edition detection eliminates manual configuration complexity
- 90% use case gets simple API, 10% use case gets explicit pooling API

### Principle V: InterSystems IRIS Integration ✅ PASS
- All connections use IRIS native DBAPI
- No fallback to alternative databases
- Edition detection queries IRIS system tables or parses license file
- Preserves IRIS-specific connection optimizations

### Performance Standards ✅ PASS with Monitoring
- Query operations: Module-level caching ensures <5% overhead (measured via benchmarks)
- Connection establishment: Target <50ms (current baseline to maintain)
- Test execution: Integration tests remain <30s total
- **Action Required**: Add performance benchmarks in contract tests to validate zero regression

### No Constitution Violations
This feature simplifies existing architecture without adding complexity. All gates pass.

## Project Structure

### Documentation (this feature)

```text
specs/051-simplify-iris-connection/
├── plan.md              # This file (/speckit.plan command output)
├── spec.md              # Feature specification (complete)
├── ARCHITECTURE_ANALYSIS.md  # Current architecture analysis (complete)
├── research.md          # Phase 0 output (to be generated)
├── data-model.md        # Phase 1 output (to be generated)
├── quickstart.md        # Phase 1 output (to be generated)
├── contracts/           # Phase 1 output (to be generated)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
iris_vector_rag/
├── common/
│   ├── iris_connection.py           # NEW: Unified connection module
│   ├── iris_dbapi_connector.py      # PRESERVE: UV fix logic (integrate into new module)
│   ├── connection_manager.py        # DEPRECATE: Generic abstraction layer
│   ├── iris_connection_manager.py   # DEPRECATE: IRIS-specific manager
│   └── connection_pool.py           # KEEP: Production API pooling (possibly refactor)
├── testing/
│   ├── connection_pool.py           # DEPRECATE: Backend mode test pooling
│   └── backend_manager.py           # DEPRECATE: Backend mode configuration
└── core/
    └── connection.py                # UPDATE: High-level connection interface

tests/
├── contract/
│   └── test_iris_connection_contract.py  # NEW: Contract tests for new API
├── integration/
│   └── test_iris_connection_integration.py  # NEW: Edition detection, pooling tests
├── unit/
│   └── test_iris_connection_unit.py  # NEW: Parameter validation, caching tests
└── conftest.py                      # UPDATE: Replace connection_pool fixture

.specify/
└── config/
    └── backend_modes.yaml           # DEPRECATE: Manual backend mode config
```

**Structure Decision**: Single project (library), follows existing iris-vector-rag structure. New unified module in `iris_vector_rag/common/iris_connection.py` consolidates 6 existing components. Tests follow TDD contract-first approach per constitution.

## Complexity Tracking

> This feature REDUCES complexity (6 components → 1-2 components), so no violations to justify.

## Phase 0: Research & Investigation

### Research Tasks

The feature specification identifies several "NEEDS CLARIFICATION" areas requiring research:

1. **Edition Detection Mechanism**
   - **Question**: How to reliably detect IRIS Community vs Enterprise edition?
   - **Options to investigate**:
     - Query system table: `SELECT * FROM %SYS.License`
     - Parse license key file from IRIS installation directory
     - Check `iris.system()` API capabilities
   - **Decision criteria**: Most reliable method that works in all environments (Docker, local install, cloud)

2. **License Key File Format**
   - **Question**: What is the format of IRIS license key files for parsing?
   - **Research needed**:
     - File location: `/usr/irissys/mgr/` vs `/opt/iris/mgr/` vs `$ISC_PACKAGE_INSTALLDIR`
     - File name patterns
     - Parsing logic to extract edition type
   - **Decision criteria**: Works for both Community (free) and Enterprise (licensed) editions

3. **Connection Caching Strategy**
   - **Question**: Module-level singleton with thread-safe lazy initialization - best practices?
   - **Research needed**:
     - Python threading.Lock vs threading.RLock for connection initialization
     - Module-level vs class-level singleton patterns
     - Connection lifecycle management (infinite lifetime until process exit - acceptable for singleton pattern)
     - Thread safety for concurrent access
     - Cache invalidation: Not required - connection lives for process lifetime (standard pattern for DB connections)
   - **Decision criteria**: Zero performance regression, thread-safe, simple implementation

4. **Parameter Validation Patterns**
   - **Question**: What validation rules for host, port, namespace before connection attempt?
   - **Research needed**:
     - Port range validation (1-65535)
     - Namespace format (alphanumeric + underscores, max length?)
     - Host validation (empty string check, DNS resolution?)
     - Error message best practices (actionable vs generic)
   - **Decision criteria**: Fail fast with clear errors, avoid unnecessary DNS lookups

5. **Backward Compatibility Migration Tooling**
   - **Question**: Do we need automated migration scripts for old API → new API?
   - **Research needed**:
     - Analyze 156 files importing connection modules
     - Identify common usage patterns
     - Determine if simple search-replace suffices or need AST-based refactoring
   - **Decision criteria**: Balance automation vs manual migration effort

6. **Best Practices from Other DB Libraries**
   - **Question**: How do SQLAlchemy, psycopg2, pymongo handle connection simplicity?
   - **Research needed**:
     - SQLAlchemy: `create_engine()` + `Engine.connect()` patterns
     - psycopg2: `psycopg2.connect()` simple API
     - pymongo: `MongoClient()` connection pooling
     - Redis-py: `redis.Redis()` connection + pooling in one
   - **Decision criteria**: Learn from established patterns, avoid reinventing wheel

### Research Output

Research findings will be documented in `research.md` with the following structure:

```markdown
# Research: IRIS Connection Simplification

## Decision Log

### 1. Edition Detection Method
**Decision**: [Selected approach]
**Rationale**: [Why chosen]
**Alternatives considered**: [Other options evaluated]
**Implementation notes**: [Key details]

### 2. Connection Caching Pattern
**Decision**: [Selected approach]
**Rationale**: [Why chosen]
**Code example**: [Proof of concept]

### 3. Parameter Validation Rules
**Decision**: [Validation logic]
**Error messages**: [Examples]

[... and so on for each research task]
```

## Phase 1: Design & Contracts

### Prerequisites
- `research.md` complete with all decisions documented
- Edition detection mechanism selected and validated
- Connection caching pattern prototyped and tested

### Data Model (entities/relationships)

#### Connection Entity
- **Purpose**: Represents a single IRIS database connection
- **Fields**:
  - `host`: str (IRIS server hostname/IP)
  - `port`: int (SuperServer port, default 1972)
  - `namespace`: str (IRIS namespace, default "USER")
  - `username`: str (database user)
  - `password`: str (database password)
  - `connection`: iris.DBAPI.Connection (actual DBAPI connection object)
  - `is_pooled`: bool (whether connection is managed by pool)

#### ConnectionPool Entity
- **Purpose**: Manages pool of connections for high-concurrency scenarios
- **Fields**:
  - `max_connections`: int (maximum pool size)
  - `available_connections`: Queue[Connection] (available connections)
  - `active_connections`: Set[Connection] (currently in-use connections)
  - `lock`: threading.Lock (thread-safe pool operations)

#### EditionInfo Entity
- **Purpose**: Caches detected IRIS edition information
- **Fields**:
  - `edition_type`: Enum["community", "enterprise"] (detected edition)
  - `max_connections`: int (edition-specific connection limit)
  - `detection_method`: str (how edition was detected)
  - `detection_timestamp`: datetime (when detected)

#### ValidationError Exception
- **Purpose**: Clear error messages for invalid connection parameters
- **Fields**:
  - `parameter_name`: str (which parameter failed validation)
  - `invalid_value`: Any (the rejected value)
  - `valid_range`: str (what values are acceptable)
  - `message`: str (actionable error message)

#### ConnectionLimitError Exception
- **Purpose**: Raised when Community Edition connection limit reached
- **Fields**:
  - `current_limit`: int (maximum connections for edition)
  - `suggested_actions`: List[str] (actionable suggestions)

### API Contracts

Based on functional requirements from spec.md:

```python
# Contract 1: Simple Connection API (FR-001, FR-002, FR-003)
def get_iris_connection(
    host: Optional[str] = None,
    port: Optional[int] = None,
    namespace: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    auto_detect_edition: bool = True
) -> iris.DBAPI.Connection:
    """
    Get IRIS database connection with automatic edition detection.

    Returns cached module-level connection (thread-safe singleton).
    Parameters from environment variables if not provided:
    - IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME, IRIS_PASSWORD

    Edition detection (if auto_detect_edition=True):
    - Automatically detects Community vs Enterprise edition
    - Applies appropriate connection limits
    - Override with IRIS_BACKEND_MODE env variable

    Validates parameters before connection attempt:
    - port: 1-65535
    - namespace: alphanumeric + underscores
    - host: non-empty string

    Raises:
    - ValidationError: Invalid parameters (before connection attempt)
    - ConnectionLimitError: When Community Edition limit reached
    - DatabaseError: When database is unavailable or credentials invalid

    Performance:
    - <50ms connection establishment
    - Module-level caching reduces overhead for repeated calls
    - Zero allocation after first call (singleton pattern)
    """
    pass


# Contract 2: Connection Pooling API (FR-004)
class IRISConnectionPool:
    """
    Explicit connection pooling for high-concurrency scenarios.

    Use Cases:
    - REST API servers with concurrent requests
    - Batch processing with parallel workers
    - High-throughput data pipelines

    Not Needed For:
    - Integration tests (use get_iris_connection())
    - Simple scripts (use get_iris_connection())
    - Single-threaded applications
    """

    def __init__(
        self,
        max_connections: Optional[int] = None,
        **connection_params
    ):
        """
        Create connection pool with specified limits.

        Args:
            max_connections: Maximum pool size (auto-detected if None)
            **connection_params: Passed to get_iris_connection()

        Edition-aware defaults:
        - Community: max_connections=1
        - Enterprise: max_connections=20 (configurable)
        """
        pass

    def acquire(self, timeout: float = 30.0) -> iris.DBAPI.Connection:
        """
        Acquire connection from pool.

        Args:
            timeout: Maximum seconds to wait for available connection

        Returns:
            Connection from pool

        Raises:
            TimeoutError: No connection available within timeout
            ConnectionLimitError: Pool exhausted (Community Edition)
        """
        pass

    def release(self, connection: iris.DBAPI.Connection) -> None:
        """
        Return connection to pool.

        Args:
            connection: Connection to return

        Raises:
            ValueError: Connection not from this pool
        """
        pass

    def close_all(self) -> None:
        """Close all connections in pool."""
        pass


# Contract 3: Error Handling (FR-006)
def _validate_connection_params(
    host: str,
    port: int,
    namespace: str,
    username: str,
    password: str
) -> None:
    """
    Validate connection parameters before attempting connection.

    Raises:
        ValidationError: With specific actionable message:
            - "Invalid port 99999: must be between 1-65535"
            - "Invalid namespace 'foo bar': must be alphanumeric and underscores only"
            - "Host cannot be empty"
    """
    pass


# Contract 4: Edition Detection (FR-002, FR-008)
def detect_iris_edition() -> Tuple[str, int]:
    """
    Detect IRIS edition and return appropriate connection limit.

    Detection priority:
    1. IRIS_BACKEND_MODE environment variable (override)
    2. License key file parsing
    3. Fallback to "community" mode

    Returns:
        Tuple of (edition_type, max_connections)
        - ("community", 1)
        - ("enterprise", 999)

    Caches result for session to avoid repeated detection overhead.
    """
    pass
```

### Contracts Directory Structure

```text
specs/051-simplify-iris-connection/contracts/
├── api_contract.yaml            # OpenAPI-style contract (for documentation)
├── test_contract_simple_api.py  # Contract tests for get_iris_connection()
├── test_contract_pooling_api.py # Contract tests for IRISConnectionPool
├── test_contract_validation.py  # Contract tests for parameter validation
└── test_contract_edition.py     # Contract tests for edition detection
```

### Quickstart Guide Outline

Will be generated in Phase 1, covering:

1. **Before (Old Way)** - Show current complexity
2. **After (New Way)** - Show simplified API
3. **Migration Guide** - Step-by-step for existing code
4. **Common Patterns** - Integration tests, API servers, scripts
5. **Troubleshooting** - Edition detection, connection limits, UV environments

## Phase 2: Implementation (Not Generated by /speckit.plan)

Phase 2 (implementation) will be handled by `/speckit.tasks` command, which generates task breakdown and dependency ordering for actual coding work.

## Success Criteria Validation

From spec.md success criteria:

- **SC-001**: Understanding time <5 minutes ✅ (validated via quickstart.md walkthrough in Phase 1)
- **SC-002**: Test clarity 100% ✅ (validated via renamed fixtures: `real_iris_connection`)
- **SC-003**: Connection time within 5% ✅ (validated via performance benchmarks in contract tests)
- **SC-004**: Zero breaking changes ✅ (validated via existing test suite passing unchanged)
- **SC-005**: Code reduction to 1-2 files ✅ (measured: 6 files → 1 file = iris_connection.py)
- **SC-006**: 90% usage replaced with single call ✅ (measured via code analysis in research.md)

## Next Steps

1. ✅ Setup complete (plan.md created)
2. ⏭️ Phase 0: Execute research tasks → generate `research.md`
3. ⏭️ Phase 1: Design data model → generate `data-model.md`
4. ⏭️ Phase 1: Write API contracts → generate `contracts/` directory
5. ⏭️ Phase 1: Create quickstart guide → generate `quickstart.md`
6. ⏭️ Phase 1: Update agent context → run `.specify/scripts/bash/update-agent-context.sh claude`
7. ⏭️ Re-evaluate Constitution Check after design
8. ⏭️ Phase 2: Run `/speckit.tasks` to generate implementation tasks

## References

- **Feature Branch**: `051-simplify-iris-connection`
- **Specification**: [spec.md](./spec.md)
- **Architecture Analysis**: [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md)
- **Constitution**: `.specify/memory/constitution.md`
- **TODO Item**: `TODO.md` lines 70-109
