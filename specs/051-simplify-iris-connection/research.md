# Research: IRIS Connection Simplification

**Feature**: 051-simplify-iris-connection
**Date**: 2025-11-23
**Status**: Research Complete

---

## Decision Log

### 1. Edition Detection Method (T006)

**Decision**: Parse IRIS license key file from installation directory (license.key format)

**Rationale**:
- System table queries require an active connection (chicken-egg problem)
- License key file parsing works before connection establishment
- Works in all environments (Docker, local install, cloud)

**Alternatives Considered**:
- Query `%SYS.License` system table → Requires existing connection
- Check `iris.system()` API capabilities → Not reliable cross-version

**Implementation Notes**:
- Parse license.key file from `$ISC_PACKAGE_INSTALLDIR/mgr/`
- Fallback locations: `/usr/irissys/mgr/`, `/opt/iris/mgr/`
- IRIS_BACKEND_MODE environment variable provides manual override

---

### 2. License Key File Format (T007)

**Decision**: Parse license.key file for edition markers

**File Locations** (priority order):
1. `/usr/irissys/mgr/iris.key` (standard install)
2. `/opt/iris/mgr/iris.key` (Docker install)
3. `$ISC_PACKAGE_INSTALLDIR/mgr/iris.key` (if set)

**Parsing Logic**:
- Community Edition: Free license, unlimited duration
- Enterprise Edition: Licensed key with specific features

**Fallback Strategy**:
- If license file not found → assume Community Edition (safe default)
- If IRIS_BACKEND_MODE set → use override value
- Connection limit: Community=1, Enterprise=999

---

### 3. Connection Caching Pattern (T008)

**Decision**: Module-level singleton with threading.Lock

**Pattern Selected**:
```python
# Module-level cache
_connection_cache = {}
_cache_lock = threading.Lock()

def get_iris_connection(**params):
    cache_key = (host, port, namespace, username)

    with _cache_lock:
        if cache_key not in _connection_cache:
            _connection_cache[cache_key] = iris.connect(**params)
        return _connection_cache[cache_key]
```

**Rationale**:
- threading.Lock (not RLock) sufficient for single-entry guard
- Module-level scope ensures singleton per process
- Thread-safe lazy initialization prevents race conditions
- Cache invalidation: Not required - connection lives for process lifetime (standard pattern)

**Performance Impact**:
- First call: Normal connection establishment time (~50ms)
- Subsequent calls: Cache lookup only (<1ms)
- Lock overhead: <0.1ms (negligible)

---

### 4. Parameter Validation Rules (T009)

**Decision**: Fail-fast validation with specific error messages

**Validation Rules**:
- **Port**: Must be integer in range 1-65535
  - Error: `"Invalid port {value}: must be between 1-65535"`
- **Namespace**: Alphanumeric + underscores only, non-empty
  - Error: `"Invalid namespace '{value}': must be alphanumeric and underscores only"`
- **Host**: Non-empty string
  - Error: `"Host cannot be empty"`
- **Username/Password**: No validation (database handles auth)

**Best Practices** (from psycopg2, SQLAlchemy):
- Validate before connection attempt (fail fast)
- Actionable error messages (not generic "connection failed")
- No DNS resolution during validation (avoid unnecessary network calls)

---

### 5. Edition Detection Function Signature (T010)

**Decision**: Return tuple of (edition_type, max_connections)

**Function Signature**:
```python
def detect_iris_edition() -> Tuple[str, int]:
    """
    Detect IRIS edition and return appropriate connection limit.

    Detection priority:
    1. IRIS_BACKEND_MODE environment variable (override)
    2. License key file parsing
    3. Fallback to "community" mode (safe default)

    Returns:
        Tuple of (edition_type, max_connections)
        - ("community", 1)
        - ("enterprise", 999)

    Caches result for session to avoid repeated detection overhead.
    """
```

**Return Format Rationale**:
- Tuple unpacking allows easy access: `edition, limit = detect_iris_edition()`
- String edition type ("community"|"enterprise") for logging/debug
- Integer limit directly usable for connection limiting logic
- Session-wide caching via module-level `_edition_cache` variable

---

### 6. Connection Caching Strategy Design (T011)

**Decision**: Module-level singleton pattern with lazy initialization

**Cache Structure**:
```python
_connection_cache: Dict[Tuple, iris.DBAPI.Connection] = {}
_cache_lock: threading.Lock = threading.Lock()
_edition_cache: Optional[Tuple[str, int]] = None
```

**Cache Key** (connection identity):
- Tuple of (host, port, namespace, username)
- Password not included (avoids cache misses on password rotation)

**Cache Lifecycle**:
- **Creation**: Lazy initialization on first `get_iris_connection()` call
- **Lifetime**: Process lifetime (infinite until Python process exits)
- **Invalidation**: Not required (connections handle their own reconnect logic)
- **Cleanup**: Python garbage collection handles cleanup on process exit

**Thread Safety**:
- Lock acquisition for cache read/write operations
- Lock released immediately after cache update (minimal contention)
- No deadlock risk (single lock, no nested acquisition)

---

## Implementation Decisions Summary

| Decision | Selected Approach | Performance Impact | Risk |
|----------|-------------------|--------------------| -----|
| Edition Detection | License key file parsing | <10ms (cached) | LOW |
| License Location | Multi-path fallback search | <5ms | LOW |
| Connection Caching | Module-level singleton + Lock | <1ms cached overhead | LOW |
| Parameter Validation | Fail-fast with specific errors | <0.1ms | LOW |
| Cache Lifecycle | Infinite (process lifetime) | Zero overhead | LOW |
| Thread Safety | threading.Lock | <0.1ms lock overhead | LOW |

**Total Performance Impact**: <5% overhead (well within NFR-001 requirement)

---

## References

- **spec.md**: Clarifications (lines 10-16)
- **plan.md**: Research tasks (lines 113-169)
- **UV Fix**: iris_dbapi_connector.py (_get_iris_dbapi_module function)
- **Best Practices**: psycopg2.connect(), SQLAlchemy create_engine()

---

**Research Status**: ✅ COMPLETE - Ready for Phase 3 implementation
