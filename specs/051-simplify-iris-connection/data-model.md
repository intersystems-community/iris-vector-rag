# Data Model: IRIS Connection Architecture

**Feature**: 051-simplify-iris-connection
**Date**: 2025-11-23

---

## Core Entities

### Connection Entity

**Purpose**: Represents a single IRIS database connection managed by the system

**Fields**:
- `host`: str - IRIS server hostname or IP address
- `port`: int - SuperServer port (default: 1972)
- `namespace`: str - IRIS namespace (default: "USER")
- `username`: str - Database user for authentication
- `password`: str - Database password for authentication
- `connection`: iris.DBAPI.Connection - Actual DBAPI connection object
- `is_pooled`: bool - Whether connection is managed by connection pool

**Lifecycle**:
- **Creation**: via `iris.connect(**params)`
- **Storage**: Module-level cache dict `_connection_cache`
- **Lifetime**: Process lifetime (infinite until exit)
- **Cleanup**: Python garbage collection on process termination

**Cache Key**: `Tuple[host, port, namespace, username]` (password excluded)

---

### EditionInfo Entity

**Purpose**: Caches detected IRIS edition information for session-wide reuse

**Fields**:
- `edition_type`: Literal["community", "enterprise"] - Detected edition
- `max_connections`: int - Edition-specific connection limit
  - Community: 1
  - Enterprise: 999
- `detection_method`: str - How edition was detected
  - "env_override" - IRIS_BACKEND_MODE environment variable
  - "license_file" - Parsed from license.key file
  - "default_fallback" - Safe default (community)
- `detection_timestamp`: datetime - When detection occurred

**Lifecycle**:
- **Creation**: First call to `detect_iris_edition()`
- **Storage**: Module-level variable `_edition_cache: Optional[Tuple[str, int]]`
- **Lifetime**: Session-wide (process lifetime)
- **Invalidation**: None (edition doesn't change during process lifetime)

---

### ConnectionPool Entity

**Purpose**: Manages pool of connections for high-concurrency scenarios (optional)

**Fields**:
- `max_connections`: int - Maximum pool size (edition-aware default)
- `available_connections`: queue.Queue[Connection] - Available connections
- `active_connections`: Set[Connection] - Currently in-use connections
- `lock`: threading.Lock - Thread-safe pool operations
- `connection_params`: Dict[str, Any] - Parameters for creating new connections

**Methods**:
- `__init__(max_connections, **connection_params)` - Initialize pool
- `acquire(timeout=30.0)` - Acquire connection from pool
- `release(connection)` - Return connection to pool
- `close_all()` - Close all connections in pool
- `__enter__()` / `__exit__()` - Context manager support

**Lifecycle**:
- **Creation**: Explicit instantiation via `IRISConnectionPool()`
- **Lifetime**: Until `close_all()` called or context manager exits
- **Cleanup**: Explicit via `close_all()` or context manager

---

## Exception Entities

### ValidationError Exception

**Purpose**: Raised when connection parameters fail validation (before connection attempt)

**Fields**:
- `parameter_name`: str - Which parameter failed validation
- `invalid_value`: Any - The rejected value
- `valid_range`: str - What values are acceptable
- `message`: str - Actionable error message for debugging

**Base Class**: `ValueError`

**Usage Examples**:
```python
raise ValidationError(
    parameter_name="port",
    invalid_value=99999,
    valid_range="1-65535",
    message="Invalid port 99999: must be between 1-65535"
)
```

---

### ConnectionLimitError Exception

**Purpose**: Raised when Community Edition connection limit reached

**Fields**:
- `current_limit`: int - Maximum connections for edition (typically 1)
- `suggested_actions`: List[str] - Actionable suggestions for user
  1. "Use connection queuing with IRISConnectionPool"
  2. "Run tests serially with pytest -n 0"
  3. "Set IRIS_BACKEND_MODE=community explicitly"
- `message`: str - Full error message with context

**Base Class**: `RuntimeError`

**Usage Example**:
```python
raise ConnectionLimitError(
    current_limit=1,
    suggested_actions=[...],
    message="IRIS Community Edition connection limit (1) reached. Consider: ..."
)
```

---

## Module-Level State

### Connection Cache

```python
_connection_cache: Dict[Tuple[str, int, str, str], iris.DBAPI.Connection] = {}
```

**Key**: `(host, port, namespace, username)`
**Value**: Cached IRIS DBAPI connection
**Thread Safety**: Protected by `_cache_lock`

### Edition Cache

```python
_edition_cache: Optional[Tuple[str, int]] = None
```

**Value**: `(edition_type, max_connections)` or `None` if not yet detected
**Thread Safety**: Read-only after first write (no lock needed for subsequent reads)

### Cache Lock

```python
_cache_lock: threading.Lock = threading.Lock()
```

**Purpose**: Guards `_connection_cache` read/write operations
**Scope**: Module-level singleton

---

## Relationships Diagram

```
┌─────────────────────────────────────┐
│   get_iris_connection()             │
│   (Main API Function)               │
└────────────┬────────────────────────┘
             │
             ├──> _validate_connection_params()
             │    └──> raises ValidationError
             │
             ├──> detect_iris_edition()
             │    ├──> returns (edition_type, max_connections)
             │    └──> caches in _edition_cache
             │
             └──> _connection_cache lookup
                  ├──> Cache hit: return cached connection
                  └──> Cache miss: iris.connect() + cache
                       └──> raises ConnectionLimitError if limit reached

┌──────────────────────────────────────┐
│   IRISConnectionPool                 │
│   (Optional High-Concurrency API)    │
└────────────┬─────────────────────────┘
             │
             ├──> __init__()
             │    └──> calls detect_iris_edition() for default max_connections
             │
             ├──> acquire()
             │    ├──> Queue.get() from available_connections
             │    └──> iris.connect() if pool empty
             │
             └──> release()
                  └──> Queue.put() to available_connections
```

---

## Data Flow

### Basic Connection Request

```
User Code
   ↓
get_iris_connection(host, port, namespace, username, password)
   ↓
1. Validate parameters → ValidationError if invalid
2. Detect edition → (edition, limit)
3. Check cache → Return if exists
4. Check limit → ConnectionLimitError if exceeded
5. iris.connect(**params)
6. Cache connection
7. Return connection
   ↓
User Code (with connection)
```

### Pooled Connection Request

```
User Code
   ↓
pool = IRISConnectionPool(max_connections=10)
   ↓
with pool.acquire() as conn:
    # Use connection
   ↓
Automatic release on context exit
```

---

## Persistence & State Management

**Persistence**: None - all state is in-memory, process-lifetime

**State Recovery**: Not applicable - connections re-establish on process restart

**Concurrency Model**: Thread-safe via threading.Lock guards

**Scalability**:
- Single connection: Module-level singleton (zero overhead after first call)
- Multiple connections: IRISConnectionPool with configurable limits

---

## References

- **spec.md**: Key Entities section (lines 135-147)
- **plan.md**: Data Model section (lines 204-246)
- **research.md**: Implementation decisions

---

**Status**: ✅ COMPLETE - Data model documented and ready for implementation
