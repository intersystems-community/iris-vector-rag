# IRIS Connection Architecture Analysis

**Date**: 2025-11-23
**Feature**: 051-simplify-iris-connection
**Status**: Specification Complete

## Executive Summary

Current IRIS connection architecture has **6 separate components** across multiple layers, causing significant developer confusion and onboarding friction. This analysis maps the current complexity and proposes simplification to **1-2 components**.

**Key Finding**: 156 files import connection-related modules, but most just need a simple "give me a connection" API.

## Current Architecture (6 Components)

### 1. iris_dbapi_connector.py
**Location**: `iris_vector_rag/common/iris_dbapi_connector.py`
**Lines of Code**: ~250
**Purpose**: Low-level IRIS DBAPI module import with UV environment fallback
**Key Function**: `get_iris_dbapi_connection()`
**Critical**: Contains UV compatibility fix (Issue #5) - MUST preserve

```python
# Current API
from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection

conn = get_iris_dbapi_connection()
```

### 2. ConnectionManager (Generic Abstraction)
**Location**: `iris_vector_rag/common/connection_manager.py`
**Lines of Code**: ~150
**Purpose**: High-level connection management with JDBC/ODBC/DBAPI fallback
**Key Class**: `ConnectionManager(connection_type="odbc")`

**Issues**:
- Unnecessary abstraction for most use cases
- Fallback logic adds complexity
- Overlaps with IRISConnectionManager

```python
# Current API
from iris_vector_rag.common.connection_manager import ConnectionManager

manager = ConnectionManager(connection_type="dbapi")
conn = manager.connect()
```

### 3. IRISConnectionManager (IRIS-Specific)
**Location**: `iris_vector_rag/common/iris_connection_manager.py`
**Lines of Code**: ~200
**Purpose**: DBAPI-first connection with automatic environment detection
**Key Class**: `IRISConnectionManager(prefer_dbapi=True)`

**Issues**:
- Duplicates functionality from ConnectionManager
- DBAPI vs JDBC priority adds confusion
- Environment detection complexity

```python
# Current API
from iris_vector_rag.common.iris_connection_manager import IRISConnectionManager

manager = IRISConnectionManager(prefer_dbapi=True)
conn = manager.get_connection()
```

### 4. IRISConnectionPool (API/Production)
**Location**: `iris_vector_rag/common/connection_pool.py`
**Lines of Code**: ~250
**Purpose**: Thread-safe connection pooling for REST API server
**Key Class**: `IRISConnectionPool(pool_size=20, max_overflow=10)`

**Issues**:
- Designed for production API, but concept bleeds into tests
- Most tests don't need real pooling

```python
# Current API (production)
from iris_vector_rag.common.connection_pool import IRISConnectionPool

pool = IRISConnectionPool(
    host="localhost",
    port=1972,
    namespace="USER",
    username="user",
    password="pass",
    pool_size=20,
    max_overflow=10
)

with pool.acquire() as conn:
    # Use connection
    pass
```

### 5. ConnectionPool (Testing - Backend Mode)
**Location**: `iris_vector_rag/testing/connection_pool.py`
**Lines of Code**: ~150
**Purpose**: Mode-aware connection pooling for tests (Community vs Enterprise)
**Key Class**: `ConnectionPool(mode=BackendMode.COMMUNITY)`

**Issues**:
- Semaphore-based pooling is a mock - doesn't provide real connections
- Confusing name: "pool" but it's really just a limit enforcer
- Test fixtures named `connection_pool` but don't actually pool

```python
# Current API (testing)
from iris_vector_rag.testing.connection_pool import ConnectionPool
from iris_vector_rag.testing.backend_manager import BackendMode

pool = ConnectionPool(mode=BackendMode.COMMUNITY)  # max 1 connection
with pool.acquire(timeout=30.0) as conn:
    # conn is a mock - actual connection comes from iris_dbapi_connector
    pass
```

### 6. Backend Mode Configuration
**Location**: `.specify/config/backend_modes.yaml` + loading utilities
**Purpose**: Configure Community (1 connection) vs Enterprise (999 connections) limits

**Issues**:
- Adds configuration layer that should be automatic
- Edition detection should query database, not config file
- Forces manual configuration

```yaml
# .specify/config/backend_modes.yaml
mode: community
max_connections: 1
```

## Test Fixture Confusion

**Problem**: Test fixture named `connection_pool` doesn't actually provide pooled connections.

```python
# tests/conftest.py
@pytest.fixture(scope="session")
def connection_pool(backend_configuration):
    """Creates ConnectionPool with mode-appropriate limits"""
    from iris_vector_rag.testing.connection_pool import ConnectionPool
    return ConnectionPool(mode=backend_configuration.mode)

@pytest.fixture
def iris_connection(connection_pool):
    """Acquires connection from pool"""
    with connection_pool.acquire(timeout=30.0) as conn:
        yield conn  # conn is a mock semaphore slot, not a real connection
```

**Developer Confusion**:
- "Is `iris_connection` a real connection or mocked?"
- "What does `connection_pool.acquire()` actually return?"
- "Where does the actual IRIS connection come from?"

## Usage Analysis

**Files Importing Connection Modules**: 156 files

**Breakdown by Component**:
- `iris_dbapi_connector`: ~50 files (direct low-level usage)
- `ConnectionManager`: ~30 files (generic abstraction)
- `IRISConnectionManager`: ~20 files (IRIS-specific)
- `ConnectionPool` (testing): ~40 files (pytest fixtures)
- `IRISConnectionPool` (production): ~10 files (API server only)
- Backend configuration: ~6 files (test setup)

**Key Insight**: Most usage (90%+) just needs: "give me a working IRIS connection"

## Proposed Simplified Architecture (1-2 Components)

### Component 1: iris_connection.py (Unified Module)

Single module with two APIs:
1. **Simple Connection API** (90% use case)
2. **Optional Pooling API** (10% use case - high concurrency)

```python
# New simplified API
from iris_vector_rag.common import get_iris_connection

# Simple usage (90% of cases)
conn = get_iris_connection()  # Auto-detect edition, use env vars

# With explicit params
conn = get_iris_connection(
    host="localhost",
    port=1972,
    namespace="USER",
    username="user",
    password="pass"
)

# Optional pooling (10% of cases - API server)
from iris_vector_rag.common import IRISConnectionPool

pool = IRISConnectionPool(max_connections=20)
with pool.acquire() as conn:
    # Use connection
    pass
```

**Features**:
- Automatic edition detection (Community vs Enterprise)
- Environment variable fallback
- Preserves UV compatibility fix
- Clear separation: basic vs pooling
- No configuration files required

### Component 2: Preserved UV Handling

Keep `_get_iris_dbapi_module()` function from iris_dbapi_connector.py, integrated into new module.

## Migration Strategy

### Phase 1: Create Unified Module
- Create `iris_vector_rag/common/iris_connection.py`
- Implement `get_iris_connection()` API
- Implement `IRISConnectionPool` class
- Preserve UV compatibility fix
- Add automatic edition detection

### Phase 2: Update Test Fixtures
- Replace `connection_pool` fixture with direct `get_iris_connection()`
- Rename fixtures for clarity: `iris_connection` → `real_iris_connection`
- Remove backend mode configuration dependency

### Phase 3: Gradual Migration
- Keep old APIs working (deprecation warnings)
- Update core modules first (storage, pipelines)
- Update test suite incrementally
- Update documentation and examples

### Phase 4: Deprecation
- Add clear deprecation warnings to old APIs
- Provide migration guide
- Set timeline for removal (e.g., 3 versions)

### Phase 5: Cleanup
- Remove old connection modules
- Remove backend mode configuration
- Update all imports

## Files to Modify

### New Files:
1. `iris_vector_rag/common/iris_connection.py` (new unified module)

### Files to Deprecate:
1. `iris_vector_rag/common/connection_manager.py`
2. `iris_vector_rag/common/iris_connection_manager.py`
3. `iris_vector_rag/testing/connection_pool.py`
4. `.specify/config/backend_modes.yaml`

### Files to Preserve (with integration):
1. `iris_vector_rag/common/iris_dbapi_connector.py` (UV fix logic)
2. `iris_vector_rag/common/connection_pool.py` (production API pooling)

### Test Files to Update:
1. `tests/conftest.py` (update fixtures)
2. All test files using `connection_pool` fixture (~40 files)

## Backward Compatibility Plan

**Keep Both APIs Working During Transition**:

```python
# Old API (deprecated, but works)
from iris_vector_rag.common.connection_manager import ConnectionManager
manager = ConnectionManager(connection_type="dbapi")
conn = manager.connect()
# Warning: "ConnectionManager deprecated. Use get_iris_connection() instead."

# New API (recommended)
from iris_vector_rag.common import get_iris_connection
conn = get_iris_connection()
```

**Timeline**:
- Version N: Introduce new API, deprecate old
- Version N+1: Continue supporting both, louder warnings
- Version N+2: Continue supporting both, very loud warnings
- Version N+3: Remove old APIs (breaking change, major version bump)

## Success Metrics

1. **Code Reduction**: 6 components → 1-2 components ✅
2. **File Count**: ~1100 LOC → ~300 LOC (73% reduction)
3. **Import Clarity**: Single obvious import for 90% of cases ✅
4. **Understanding Time**: <5 minutes for new developers ✅
5. **Test Clarity**: Obviously real connections vs mocks ✅
6. **Zero Breaking Changes**: During migration period ✅

## Open Questions

1. **Edition Detection**: How to reliably detect Community vs Enterprise?
   - Query system table? `SELECT * FROM %SYS.License`
   - Check license key format?
   - Configuration file fallback?

2. **Automatic Connection Limits**: Should we enforce limits automatically?
   - Community: Warn if >1 connection attempt?
   - Enterprise: No limits?
   - Or let developer manage?

3. **Pooling Strategy**: Should basic API include implicit pooling?
   - Current thinking: No - keep pooling explicit and optional
   - Rationale: Most tests/scripts don't need pooling

4. **Connection Sharing**: Should we cache connections at module level?
   - Pro: Reduces connection overhead
   - Con: Adds state management complexity
   - Recommendation: Start without caching, add if needed

## Next Steps

1. ✅ Complete specification (this document)
2. [ ] Prototype `iris_connection.py` module
3. [ ] Design edition detection mechanism
4. [ ] Write contract tests for new API
5. [ ] Implement new module with UV fix preserved
6. [ ] Update test fixtures
7. [ ] Create migration guide
8. [ ] Gradual rollout to codebase

## References

- **Feature Branch**: `051-simplify-iris-connection`
- **Specification**: `specs/051-simplify-iris-connection/spec.md`
- **TODO Item**: `TODO.md` lines 70-109
- **UV Compatibility Fix**: `iris_dbapi_connector.py:252` (commit `478d3f1b`)
- **Backend Mode Feature**: Feature 035-make-2-modes
## Baseline Metrics (Pre-Implementation - 2025-11-23)

### Connection Module Count
Total connection-related files identified: 6 components

1. iris_dbapi_connector.py (~250 LOC)
2. connection_manager.py
3. iris_connection_manager.py  
4. connection_pool.py (production)
5. testing/connection_pool.py (backend mode)
6. testing/backend_manager.py

### File Import Analysis
Files importing connection modules: 156 files
Test files using connection fixtures: ~40 files

### Performance Baseline
- Connection establishment time: Target <50ms (to be measured)
- Module overhead: <5% target for cached connections

**Status**: Baseline documented for comparison post-implementation
