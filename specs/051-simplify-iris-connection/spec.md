# Feature Specification: Simplify IRIS Connection Architecture

**Feature Branch**: `051-simplify-iris-connection`
**Created**: 2025-11-23
**Status**: Draft
**Input**: Simplify IRIS Connection/Testing/Config Architecture - reduce from 4-5 abstraction layers (ConnectionManager, ConnectionPool, iris_dbapi_connector, Backend Mode config) to 1-2 clear components. Current complexity causes developer onboarding friction and confusing test fixtures (mock pooling vs real connections). Goal: developer can understand connection flow in <5 minutes, integration tests trivially obvious, maintain backward compatibility, zero performance regression.

## Clarifications

### Session 2025-11-23

- Q: How should the system detect Community vs Enterprise edition for automatic connection limiting? → A: Parse IRIS license key file from installation directory (license.key format)
- Q: How long should the deprecated old connection APIs be supported during migration? → A: 1 major version (fast deprecation cycle, remove in next breaking release)
- Q: When Community Edition connection limit is reached, what should the system do? → A: Raise ConnectionLimitError exception with message suggesting connection queuing or serial test execution
- Q: Should `get_iris_connection()` cache connections at the module level to reduce overhead? → A: Module-level singleton with thread-safe lazy initialization
- Q: When should connection parameters (host, port, namespace) be validated? → A: Validate on first call to get_iris_connection(), raise early with specific errors before attempting connection

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Direct Connection Usage (Priority: P1)

**As a** developer writing integration tests
**I want** to get an IRIS database connection with a single, obvious function call
**So that** I don't need to understand multiple abstraction layers or wonder if I'm using a mock

**Why this priority**: This is the most common use case - 90% of developers just need a simple "give me a connection" API. Simplifying this removes the biggest barrier to understanding the codebase.

**Independent Test**: Can be fully tested by writing a simple test that calls the new connection API and executes a SQL query. Delivers immediate value by making the most common case trivial.

**Acceptance Scenarios**:

1. **Given** a developer needs to write an integration test
   **When** they call `get_iris_connection()` with connection params
   **Then** they receive a working IRIS connection without needing to understand pooling, backend modes, or other abstractions

2. **Given** a developer is reading test code
   **When** they see `get_iris_connection()` in a test
   **Then** it's immediately obvious this is a real database connection (not mocked)

3. **Given** a new developer joins the project
   **When** they read connection usage examples
   **Then** they can understand the full connection flow in under 5 minutes

---

### User Story 2 - Automatic Backend Mode Handling (Priority: P2)

**As a** developer running tests
**I want** the connection system to automatically handle Community vs Enterprise edition differences
**So that** I don't need to manually configure backend modes or worry about license pool exhaustion

**Why this priority**: Backend mode configuration adds significant complexity. Automating this eliminates a common source of confusion and test failures.

**Independent Test**: Can be tested by running tests against both Community and Enterprise editions and verifying correct connection limits are applied automatically.

**Acceptance Scenarios**:

1. **Given** tests are running against IRIS Community Edition
   **When** connection system detects the edition
   **Then** it automatically limits connections to prevent license pool exhaustion

2. **Given** tests are running against IRIS Enterprise Edition
   **When** connection system detects the edition
   **Then** it allows parallel connections for faster test execution

3. **Given** a developer needs to override edition detection
   **When** they set `IRIS_BACKEND_MODE` environment variable
   **Then** the system respects the override

---

### User Story 3 - Clear Connection Pooling (Priority: P3)

**As a** developer working with high-concurrency scenarios
**I want** an explicit, optional connection pooling API
**So that** I can use pooling when needed without it being mixed into basic connection logic

**Why this priority**: Pooling is needed for specific use cases (API server, high concurrency). Most tests and simple scripts don't need it. Making it optional and explicit reduces complexity for the common case.

**Independent Test**: Can be tested by creating a connection pool and verifying multiple concurrent connections work correctly with proper resource management.

**Acceptance Scenarios**:

1. **Given** a developer needs connection pooling for an API server
   **When** they use the explicit pooling API
   **Then** they get a connection pool with clear lifecycle management

2. **Given** a developer is writing a simple script
   **When** they use the basic connection API
   **Then** they don't need to think about pooling at all

3. **Given** pooled connections are in use
   **When** connections are returned to the pool
   **Then** resources are properly cleaned up and connections are reused

---

### Edge Cases

- What happens when IRIS database is not running and connection is requested?
  - System should fail fast with clear error message indicating database unavailable

- What happens when connection parameters are invalid?
  - System validates parameters on first call to get_iris_connection() before attempting database connection
  - Raises specific ValidationError with actionable message (e.g., "Invalid port number '99999': must be between 1-65535" vs generic "connection failed")
  - Validation includes: port range (1-65535), namespace format (alphanumeric + underscores), host not empty

- What happens when maximum connections are reached in Community Edition?
  - System raises ConnectionLimitError exception with message: "IRIS Community Edition connection limit (1) reached. Consider: 1) Using connection queuing with IRISConnectionPool, 2) Running tests serially with pytest -n 0, 3) Setting IRIS_BACKEND_MODE=community explicitly."

- What happens when connection is lost mid-operation?
  - System should detect connection loss and raise appropriate exception for retry logic

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a simple `get_iris_connection()` function that returns a working IRIS database connection
- **FR-002**: System MUST automatically detect IRIS edition (Community vs Enterprise) by parsing license key file format from IRIS installation directory and apply appropriate connection limits (1 connection for Community, unlimited for Enterprise)
- **FR-003**: System MUST provide connection parameter validation before attempting connection. Validate on first call to get_iris_connection() and raise specific errors (e.g., "Invalid port number: must be 1-65535" instead of generic "connection failed").
- **FR-004**: System MUST allow optional connection pooling via explicit API (e.g., `IRISConnectionPool()`)
- **FR-005**: System MUST maintain backward compatibility with existing connection usage
- **FR-006**: System MUST provide clear error messages for common failure scenarios (database down, invalid credentials, license pool exhausted). When Community Edition connection limit reached, raise ConnectionLimitError with actionable suggestions.
- **FR-007**: System MUST handle UV environment iris module import correctly (preserve existing fix)
- **FR-008**: System MUST allow manual backend mode override via `IRIS_BACKEND_MODE` environment variable

### Non-Functional Requirements

- **NFR-001**: Connection establishment time must not increase (zero performance regression). Implement module-level connection caching with thread-safe lazy initialization to reduce overhead for repeated calls.
- **NFR-002**: Memory usage must not increase compared to current implementation
- **NFR-003**: New developer onboarding time for understanding connection flow must be under 5 minutes
- **NFR-004**: Test fixture usage must be obviously "real connection" vs "mock" from code inspection
- **NFR-005**: Migration from old API to new API must be possible incrementally (support both during transition)

### Key Entities

**Current Architecture (To Be Simplified)**:
- **ConnectionManager**: High-level connection management
- **ConnectionPool**: Connection pooling abstraction
- **iris_dbapi_connector**: Low-level IRIS DBAPI connection logic
- **Backend Mode Config**: Community vs Enterprise edition handling
- **UV Environment Handling**: iris module import fallback logic

**Target Simplified Architecture**:
- **IRISConnection**: Single unified connection interface with automatic edition detection
- **IRISConnectionPool** (optional): Explicit pooling for high-concurrency scenarios only

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: New developers can understand connection flow by reading code for under 5 minutes (measured via onboarding documentation walkthrough)
- **SC-002**: Test fixtures using real connections are obviously real (measured via developer survey: "Can you tell if this is mocked?" - target 100% correct answers)
- **SC-003**: Connection establishment time is within 5% of current implementation (measured via performance benchmarks)
- **SC-004**: Zero breaking changes for existing tests and code (measured via full test suite passing without modifications)
- **SC-005**: Reduce connection-related code from 4-5 files to 1-2 files (measured via file count before/after)
- **SC-006**: 90% of connection usage can be replaced with single `get_iris_connection()` call (measured via code analysis)

## Current Architecture Analysis

### Current Abstraction Layers (4-5 components):

1. **iris_dbapi_connector.py** (~250 lines)
   - Low-level IRIS DBAPI module import with UV environment fallback
   - Function: `get_iris_dbapi_connection()`
   - Handles: Module import edge cases, connection parameter validation

2. **ConnectionManager** (location TBD - need to find)
   - Mid-level connection lifecycle management
   - Purpose: Abstract connection creation/destruction
   - Issues: Unclear relationship to ConnectionPool

3. **ConnectionPool** (location TBD - need to find)
   - Connection pooling abstraction
   - Issues: Used in tests even when single connection is sufficient
   - Confusion: Test fixtures named "connection_pool" but provide real connections

4. **Backend Mode Configuration** (`.specify/config/backend_modes.yaml` + loading logic)
   - Handles Community vs Enterprise edition differences
   - Issues: Adds configuration layer that should be automatic
   - Files: `backend_modes.yaml`, loading utilities in test framework

5. **Test Fixtures** (pytest fixtures in conftest.py files)
   - Provide connections to tests
   - Issues: Multiple fixtures with unclear relationships
   - Confusion: `connection_pool` fixture doesn't actually pool, it provides real connections

### Problems with Current Architecture:

1. **Developer Confusion**: "Is this connection real or mocked?" requires understanding 4-5 files
2. **Onboarding Friction**: New developers spend hours understanding connection setup
3. **Test Brittleness**: Backend mode misconfiguration causes license pool exhaustion
4. **Unnecessary Abstraction**: ConnectionPool used even for single-connection scenarios
5. **Unclear Relationships**: ConnectionManager vs ConnectionPool responsibilities overlap

## Target Simplified Architecture

### Proposed Architecture (1-2 components):

1. **iris_connection.py** (unified connection module)
   ```python
   # Simple connection API (90% use case)
   def get_iris_connection(
       host: str = None,
       port: int = None,
       namespace: str = None,
       username: str = None,
       password: str = None,
       auto_detect_edition: bool = True
   ) -> iris.DBAPI.Connection:
       """Get IRIS database connection with automatic edition detection.

       Returns cached module-level connection (thread-safe singleton).

       Parameters from environment variables if not provided:
       - IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME, IRIS_PASSWORD

       Edition detection:
       - Automatically detects Community vs Enterprise edition by parsing license file
       - Applies appropriate connection limits (1 for Community, unlimited for Enterprise)
       - Override with IRIS_BACKEND_MODE env variable

       Raises:
       - ConnectionLimitError: When Community Edition limit reached
       """
       pass

   # Optional pooling API (10% use case - high concurrency)
   class IRISConnectionPool:
       """Explicit connection pooling for high-concurrency scenarios."""
       def __init__(self, max_connections: int = None, **conn_params):
           pass

       def acquire(self) -> iris.DBAPI.Connection:
           pass

       def release(self, connection: iris.DBAPI.Connection):
           pass
   ```

2. **UV Environment Handling** (preserve existing fix)
   - Keep `_get_iris_dbapi_module()` function from iris_dbapi_connector.py
   - Integrate into new iris_connection.py module

### Migration Strategy:

1. **Phase 1 (Version N)**: Create new `iris_connection.py` module with simplified API
2. **Phase 2 (Version N)**: Update test fixtures to use new API while keeping old API working with deprecation warnings
3. **Phase 3 (Version N)**: Gradually migrate existing code to new API
4. **Phase 4 (Version N+1 - Breaking)**: Remove old connection APIs (ConnectionManager, IRISConnectionManager, testing ConnectionPool) in next major version release

## Open Questions

1. **ConnectionManager Location**: Where is ConnectionManager currently implemented? (need to find and analyze)
2. **ConnectionPool Location**: Where is ConnectionPool currently implemented? (need to find and analyze)
3. **Usage Patterns**: What percentage of code uses pooling vs single connections? (need to analyze)

## Next Steps

1. Audit codebase to find ConnectionManager and ConnectionPool implementations
2. Analyze all connection usage patterns to understand requirements
3. Design edition detection mechanism (query vs configuration)
4. Create prototype of new iris_connection.py module
5. Write contract tests for new API
6. Implement migration tooling to help transition existing code
