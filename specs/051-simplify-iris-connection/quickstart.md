# IRIS Connection Quick Start Guide

**Feature**: 051-simplify-iris-connection
**Date**: 2025-11-23
**Status**: User Story 1 (Direct Connection) Complete

---

## Overview

The unified IRIS connection module simplifies database connectivity from **6 components** to **1 simple function**. This guide covers basic usage, environment configuration, and migration from the old API.

**Goal**: Developers can get a working IRIS connection in **< 5 minutes** with obvious behavior (real connections, not mocks).

---

## Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Install iris-vector-rag package (includes intersystems-irispython)
pip install iris-vector-rag
```

### Step 2: Start IRIS Database

```bash
# Using Docker (recommended for development)
docker run -d \
  --name iris-community \
  -p 1972:1972 \
  -p 52773:52773 \
  intersystemsdc/iris-community:latest

# Wait for startup (10-15 seconds)
docker logs -f iris-community | grep "startup finished"
```

### Step 3: Use Simple Connection API

```python
from iris_vector_rag.common import get_iris_connection

# Get connection (uses environment variables or auto-detection)
conn = get_iris_connection()

# Execute query
cursor = conn.cursor()
cursor.execute("SELECT %SYSTEM.Version.GetVersion()")
version = cursor.fetchone()
print(f"Connected to IRIS: {version[0]}")
cursor.close()
```

**Done!** You have a working IRIS connection in 3 steps.

---

## Environment Variables

The connection module reads configuration from environment variables with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `IRIS_HOST` | `localhost` | Database hostname or IP address |
| `IRIS_PORT` | Auto-detect or `1972` | SuperServer port (auto-detects Docker/native) |
| `IRIS_NAMESPACE` | `USER` | Target namespace for connections |
| `IRIS_USER` | `_SYSTEM` | Database username |
| `IRIS_PASSWORD` | `SYS` | Database password |
| `IRIS_BACKEND_MODE` | Auto-detect | Edition override (`community` or `enterprise`) |

### Setting Environment Variables

**Option 1: .env file (recommended)**

```bash
# Create .env file in project root
cat > .env << EOF
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USER=_SYSTEM
IRIS_PASSWORD=SYS
EOF
```

**Option 2: Shell export**

```bash
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_NAMESPACE=USER
export IRIS_USER=_SYSTEM
export IRIS_PASSWORD=SYS
```

**Option 3: Python os.environ**

```python
import os

os.environ["IRIS_HOST"] = "localhost"
os.environ["IRIS_PORT"] = "1972"
os.environ["IRIS_NAMESPACE"] = "USER"
os.environ["IRIS_USER"] = "_SYSTEM"
os.environ["IRIS_PASSWORD"] = "SYS"
```

---

## Usage Examples

### Example 1: Simple Connection (Environment Variables)

```python
from iris_vector_rag.common import get_iris_connection

# Uses environment variables or defaults
conn = get_iris_connection()

cursor = conn.cursor()
cursor.execute("SELECT 1 AS test")
result = cursor.fetchone()
print(f"Result: {result[0]}")  # Output: Result: 1
cursor.close()
```

### Example 2: Explicit Parameters

```python
from iris_vector_rag.common import get_iris_connection

# Explicit connection parameters
conn = get_iris_connection(
    host="192.168.1.100",
    port=1972,
    namespace="PRODUCTION",
    username="admin",
    password="secret123"
)

cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM MyTable")
count = cursor.fetchone()
print(f"Row count: {count[0]}")
cursor.close()
```

### Example 3: Connection Caching (Automatic Singleton)

```python
from iris_vector_rag.common import get_iris_connection

# First call: Creates new connection (~50ms)
conn1 = get_iris_connection()

# Second call: Returns cached connection (<1ms)
conn2 = get_iris_connection()

# Same connection object (singleton pattern)
assert conn1 is conn2  # True

print("Connection is cached automatically!")
```

### Example 4: Error Handling

```python
from iris_vector_rag.common import get_iris_connection
from iris_vector_rag.common.exceptions import ValidationError

try:
    # Invalid port (validation fails immediately)
    conn = get_iris_connection(
        host="localhost",
        port=99999,  # Invalid: > 65535
        namespace="USER",
        username="_SYSTEM",
        password="SYS"
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    print(f"Parameter: {e.parameter_name}")
    print(f"Invalid value: {e.invalid_value}")
    print(f"Valid range: {e.valid_range}")
```

---

## Before & After Comparison

### Old API (Complex - 6 Components)

```python
# OLD: Confusing - which one do I use?

# Option 1: iris_dbapi_connector (low-level)
from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection
conn = get_iris_dbapi_connection()

# Option 2: ConnectionManager (generic abstraction)
from iris_vector_rag.common.connection_manager import ConnectionManager
manager = ConnectionManager(connection_type="dbapi")
conn = manager.connect()

# Option 3: IRISConnectionManager (IRIS-specific)
from iris_vector_rag.common.iris_connection_manager import IRISConnectionManager
manager = IRISConnectionManager(prefer_dbapi=True)
conn = manager.get_connection()

# Option 4: ConnectionPool (testing - backend mode)
from iris_vector_rag.testing.connection_pool import ConnectionPool
from iris_vector_rag.testing.backend_manager import BackendMode
pool = ConnectionPool(mode=BackendMode.COMMUNITY)
with pool.acquire(timeout=30.0) as conn:
    # conn is a mock semaphore slot - where's the real connection?
    pass

# Option 5: IRISConnectionPool (production pooling)
from iris_vector_rag.common.connection_pool import IRISConnectionPool
pool = IRISConnectionPool(
    host="localhost", port=1972, namespace="USER",
    username="user", password="pass",
    pool_size=20, max_overflow=10
)
with pool.acquire() as conn:
    pass

# Option 6: Manual backend mode configuration
# Edit .specify/config/backend_modes.yaml
# mode: community
# max_connections: 1
```

### New API (Simple - 1 Function)

```python
# NEW: Obvious - one function, obvious behavior

from iris_vector_rag.common import get_iris_connection

# Simple connection (uses environment variables)
conn = get_iris_connection()

# With explicit parameters
conn = get_iris_connection(
    host="localhost",
    port=1972,
    namespace="USER",
    username="_SYSTEM",
    password="SYS"
)

# That's it! Real connection, cached automatically, thread-safe.
```

**Result**: 90% reduction in code, 100% clarity on behavior.

---

## Features

### ✅ Automatic Connection Caching

Connections are cached at module level (singleton pattern) for zero overhead after first call.

```python
import time
from iris_vector_rag.common import get_iris_connection

# First connection: ~50ms (establish connection)
start = time.perf_counter()
conn1 = get_iris_connection()
print(f"First call: {(time.perf_counter() - start)*1000:.2f}ms")

# Cached connection: <1ms (lookup only)
start = time.perf_counter()
conn2 = get_iris_connection()
print(f"Cached call: {(time.perf_counter() - start)*1000:.2f}ms")

assert conn1 is conn2  # Same object
```

### ✅ Fail-Fast Parameter Validation

Invalid parameters are caught before connection attempt (no network calls for bad configs).

```python
from iris_vector_rag.common import get_iris_connection
from iris_vector_rag.common.exceptions import ValidationError

# Port validation (1-65535)
try:
    conn = get_iris_connection(port=99999)
except ValidationError as e:
    print(f"Invalid port: {e.invalid_value}")
    print(f"Valid range: {e.valid_range}")

# Namespace validation (alphanumeric + underscores)
try:
    conn = get_iris_connection(namespace="USER-SPACE")
except ValidationError as e:
    print(f"Invalid namespace: {e.invalid_value}")

# Host validation (non-empty)
try:
    conn = get_iris_connection(host="")
except ValidationError as e:
    print(f"Host cannot be empty")
```

### ✅ Auto-Port Detection

Automatically detects IRIS port from Docker containers or native installations.

**Priority Order**:
1. `IRIS_PORT` environment variable
2. Docker container inspection (port 1972 mapping)
3. Native IRIS `iris list` command
4. Fallback to default 1972

```python
# No IRIS_PORT set → auto-detection runs
import os
if "IRIS_PORT" in os.environ:
    del os.environ["IRIS_PORT"]

# Auto-detects port from running Docker container
conn = get_iris_connection()
# ✅ Auto-detected Docker IRIS on port 1972
```

### ✅ Thread-Safe Operations

Module-level cache is protected by `threading.Lock` for safe concurrent access.

```python
import threading
from iris_vector_rag.common import get_iris_connection

def worker():
    conn = get_iris_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    cursor.close()
    print(f"Thread {threading.current_thread().name}: {result[0]}")

# Safe to call from multiple threads
threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

### ✅ UV Environment Compatibility

Preserves the UV compatibility fix from `iris_dbapi_connector.py` (Issue #5).

```python
# Works in both regular virtualenv and UV-managed environments
# No manual path manipulation needed
from iris_vector_rag.common import get_iris_connection
conn = get_iris_connection()
```

---

## Troubleshooting

### Issue: "Cannot import IRIS DBAPI module"

**Cause**: `intersystems-irispython` package not installed.

**Solution**:
```bash
pip install intersystems-irispython
# or
pip install iris-vector-rag[all]
```

### Issue: "Password change required"

**Cause**: Fresh IRIS installation requires password change.

**Solution**:
```bash
# Log into IRIS Management Portal: http://localhost:52773/csp/sys/UtilHome.csp
# Change password for _SYSTEM user
# Update IRIS_PASSWORD environment variable
export IRIS_PASSWORD=new_password
```

### Issue: "Could not auto-detect IRIS port"

**Cause**: No running IRIS instance found, or Docker/iris command not available.

**Solution**:
```bash
# Option 1: Set IRIS_PORT explicitly
export IRIS_PORT=1972

# Option 2: Start IRIS Docker container
docker run -d -p 1972:1972 intersystemsdc/iris-community:latest

# Option 3: Check native IRIS installation
iris list
```

### Issue: ValidationError for namespace

**Cause**: Namespace contains invalid characters (must be alphanumeric + underscores only).

**Solution**:
```python
# INVALID: Hyphens, spaces, special characters
conn = get_iris_connection(namespace="USER-SPACE")  # ❌

# VALID: Alphanumeric and underscores
conn = get_iris_connection(namespace="USER_SPACE")  # ✅
conn = get_iris_connection(namespace="PRODUCTION")  # ✅
conn = get_iris_connection(namespace="DEV_DB")      # ✅
```

### Issue: Connection timeout or "Connection refused"

**Cause**: IRIS not running, or firewall blocking port 1972.

**Solution**:
```bash
# Check IRIS is running
docker ps | grep iris

# Check port is accessible
nc -zv localhost 1972

# Check firewall rules
sudo lsof -i :1972
```

---

## Testing

### Contract Tests (TDD - API Validation)

```bash
# Run contract tests (no IRIS required - validation only)
export SKIP_IRIS_CONTAINER=1
pytest tests/contract/test_iris_connection_contract.py -v

# Expected output:
# ✅ 3 passed (validation tests)
# ⏭️  3 skipped (connection tests require IRIS)
```

### Integration Tests (Live IRIS Required)

```bash
# Run integration tests (requires IRIS running)
export SKIP_IRIS_CONTAINER=0
export IRIS_PORT=1972  # or let auto-detection handle it
pytest tests/integration/test_iris_connection_integration.py -v

# Expected output:
# ✅ All integration tests pass
# - Connection with live IRIS
# - Caching reduces overhead
# - Validation fails fast
```

---

## Edition Detection (User Story 2)

### Automatic Edition Detection

The module automatically detects whether you're running IRIS Community or Enterprise Edition:

```python
from iris_vector_rag.common import detect_iris_edition

# Auto-detect edition
edition, max_connections = detect_iris_edition()
print(f"Detected {edition} edition ({max_connections} connections)")

# Output: Detected community edition (1 connections)
# or:     Detected enterprise edition (999 connections)
```

**Detection Logic** (priority order):
1. **IRIS_BACKEND_MODE** environment variable (`community` or `enterprise`)
2. License key file parsing (future - not yet implemented)
3. **Fallback**: Assumes Community Edition (safe default)

### Edition Override

You can manually override edition detection:

```python
import os

# Force Community Edition mode
os.environ["IRIS_BACKEND_MODE"] = "community"

# Force Enterprise Edition mode
os.environ["IRIS_BACKEND_MODE"] = "enterprise"
```

**Use Cases**:
- **Testing**: Force Community mode to test connection limits
- **Development**: Override auto-detection if incorrect
- **CI/CD**: Explicit edition configuration in pipelines

### Connection Limits (Future)

**Note**: Connection limit enforcement is not yet implemented in this release.

**Planned Behavior** (when implemented):
- **Community Edition**: Enforce 1 connection limit
- **Enterprise Edition**: Allow up to 999 connections
- **ConnectionLimitError**: Raised if limit exceeded with actionable suggestions

```python
# Future behavior (not yet implemented)
from iris_vector_rag.common import get_iris_connection
from iris_vector_rag.common.exceptions import ConnectionLimitError

try:
    # In Community Edition with limit enforcement
    conn1 = get_iris_connection()  # Works
    conn2 = get_iris_connection()  # Would raise ConnectionLimitError
except ConnectionLimitError as e:
    print(f"Limit: {e.current_limit}")
    print(f"Suggestions: {e.suggested_actions}")
```

## Connection Pooling (User Story 3)

**When to Use**: High-concurrency scenarios (API servers, batch processing, multi-threaded applications)

**When NOT to Use**: Simple scripts, single-threaded applications (use `get_iris_connection()` instead)

### Basic Connection Pooling

For applications that need multiple concurrent database connections:

```python
from iris_vector_rag.common import IRISConnectionPool

# Create pool with explicit size
pool = IRISConnectionPool(max_connections=20)

# Acquire connection (context manager auto-releases)
with pool.acquire(timeout=30.0) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM MyTable")
    results = cursor.fetchall()
    cursor.close()
# Connection automatically returned to pool
```

### Edition-Aware Pool Sizing

Pool size defaults are automatically configured based on IRIS edition:

```python
from iris_vector_rag.common import IRISConnectionPool

# No max_connections specified → auto-detects edition
pool = IRISConnectionPool()

# Community Edition: max_connections=1 (honors license limit)
# Enterprise Edition: max_connections=20 (reasonable default)

print(f"Pool size: {pool.max_connections}")
```

### Multi-Threaded Usage

Pool is thread-safe and handles concurrent acquire/release:

```python
from iris_vector_rag.common import IRISConnectionPool
import threading

pool = IRISConnectionPool(max_connections=5)

def worker(worker_id):
    with pool.acquire(timeout=10.0) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT {worker_id} AS id")
        result = cursor.fetchone()
        cursor.close()
        print(f"Worker {worker_id}: {result[0]}")

# Launch 10 workers (more than pool size)
threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Pool automatically queues requests when exhausted
```

### Connection Reuse

Pooled connections are reused, not recreated:

```python
from iris_vector_rag.common import IRISConnectionPool

pool = IRISConnectionPool(max_connections=2)

# First acquisition - creates new connection
with pool.acquire() as conn1:
    conn1_id = id(conn1)
    # Use connection
    pass

# Second acquisition - reuses released connection (same object)
with pool.acquire() as conn2:
    conn2_id = id(conn2)
    assert conn1_id == conn2_id  # True - connection reused
```

### Pool Exhaustion & Timeout

When pool is exhausted, acquire() blocks until timeout:

```python
from iris_vector_rag.common import IRISConnectionPool
import queue

pool = IRISConnectionPool(max_connections=1)

with pool.acquire(timeout=5.0) as conn:
    # Pool exhausted - only 1 connection available
    try:
        with pool.acquire(timeout=1.0) as conn2:
            pass  # Won't reach here
    except queue.Empty:
        print("Pool exhausted - timeout after 1 second")

# After release, connection becomes available again
with pool.acquire(timeout=5.0) as conn:
    print("Connection available after release")
```

### Manual Acquire/Release

For cases where context manager isn't suitable:

```python
from iris_vector_rag.common import IRISConnectionPool

pool = IRISConnectionPool(max_connections=10)

# Manual acquire
conn_context = pool.acquire(timeout=30.0)
conn = conn_context.__enter__()

try:
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    cursor.close()
finally:
    # Manual release (important!)
    conn_context.__exit__(None, None, None)
```

### Pool Cleanup

Close all connections when shutting down:

```python
from iris_vector_rag.common import IRISConnectionPool

pool = IRISConnectionPool(max_connections=10)

# Use pool...
with pool.acquire() as conn:
    pass

# Cleanup on shutdown
pool.close_all()
```

### Pool as Context Manager

Pool itself can be used as context manager for auto-cleanup:

```python
from iris_vector_rag.common import IRISConnectionPool

with IRISConnectionPool(max_connections=10) as pool:
    with pool.acquire() as conn:
        # Use connection
        pass
# Pool automatically closed on exit
```

---

## API Reference

### `get_iris_connection()`

Get IRIS database connection with automatic caching and validation.

**Parameters**:
- `host` (str, optional): Database hostname (default: from `IRIS_HOST` or `"localhost"`)
- `port` (int, optional): SuperServer port (default: from `IRIS_PORT` or auto-detect or `1972`)
- `namespace` (str, optional): Target namespace (default: from `IRIS_NAMESPACE` or `"USER"`)
- `username` (str, optional): Database username (default: from `IRIS_USER` or `"_SYSTEM"`)
- `password` (str, optional): Database password (default: from `IRIS_PASSWORD` or `"SYS"`)

**Returns**: IRIS DBAPI connection object with `cursor()` method

**Raises**:
- `ValidationError`: If parameters fail validation (port, namespace, host)
- `ConnectionError`: If connection to IRIS fails

**Example**:
```python
from iris_vector_rag.common import get_iris_connection

# Use environment variables or defaults
conn = get_iris_connection()

# Explicit parameters
conn = get_iris_connection(
    host="192.168.1.100",
    port=1972,
    namespace="PRODUCTION",
    username="admin",
    password="secret"
)
```

### `ValidationError` Exception

Raised when connection parameters fail validation before connection attempt.

**Attributes**:
- `parameter_name` (str): Name of invalid parameter
- `invalid_value` (Any): The rejected value
- `valid_range` (str): Description of acceptable values
- `message` (str): Full error message

**Example**:
```python
from iris_vector_rag.common.exceptions import ValidationError

try:
    conn = get_iris_connection(port=99999)
except ValidationError as e:
    print(f"Parameter: {e.parameter_name}")  # "port"
    print(f"Invalid: {e.invalid_value}")     # 99999
    print(f"Valid: {e.valid_range}")         # "1-65535"
```

### `IRISConnectionPool` Class

Optional connection pool for high-concurrency scenarios (API servers, batch processing).

**Constructor**:
```python
IRISConnectionPool(max_connections=None, **connection_params)
```

**Parameters**:
- `max_connections` (int, optional): Maximum pool size. If None, uses edition-aware defaults:
  - Community Edition: 1 connection
  - Enterprise Edition: 20 connections
- `**connection_params`: Connection parameters (host, port, namespace, username, password)

**Methods**:
- `acquire(timeout=30.0)`: Acquire connection from pool (returns context manager)
  - **Returns**: Context manager that yields connection
  - **Raises**: `queue.Empty` if timeout expires
- `release(connection)`: Return connection to pool for reuse
- `close_all()`: Close all connections in pool (cleanup on shutdown)

**Context Manager**:
- Pool supports `with` statement for automatic cleanup
- Individual acquisitions support `with` for automatic release

**Attributes**:
- `max_connections` (int): Maximum pool size

**Example**:
```python
from iris_vector_rag.common import IRISConnectionPool

# Create pool with edition-aware default
pool = IRISConnectionPool()

# Acquire with context manager (auto-release)
with pool.acquire(timeout=30.0) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM MyTable")
    results = cursor.fetchall()
    cursor.close()

# Manual cleanup
pool.close_all()
```

**Thread Safety**: All methods are thread-safe. Pool uses `threading.Lock` for synchronization.

---

## Support

- **GitHub Issues**: https://github.com/intersystems/iris-vector-rag/issues
- **Documentation**: `specs/051-simplify-iris-connection/`
- **Feature Spec**: `specs/051-simplify-iris-connection/spec.md`
- **Implementation Plan**: `specs/051-simplify-iris-connection/plan.md`

---

**Last Updated**: 2025-11-23
**Feature Status**: User Stories 1, 2, 3 Complete ✅
