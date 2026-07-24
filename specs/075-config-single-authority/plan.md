# Plan: Config Single Authority

## Overview

Consolidate IRIS connection parameters to a single authority: `ConfigurationManager`.
Currently three sources conflict (defaults YAML, manager.py hardcodes, iris_connection.py env vars).
This plan makes `ConfigurationManager` the single source of truth, enforces precedence
(env > YAML > hardcoded defaults), and wires `ConnectionManager` to use it.

## Key Files

- `iris_vector_rag/config/default_config.yaml:6` — Fix port from "1974" to `1972` (int)
- `iris_vector_rag/config/manager.py:521` — Fix hardcoded port from `1974` to `1972`
- `iris_vector_rag/core/connection.py:55–63` — Refactor `get_connection()` to pass resolved
  params to `get_iris_connection()`
- `iris_vector_rag/common/iris_connection.py:110–122` — Already accepts explicit params;
  no changes needed (already correct per FR-003)
- `Makefile` — Expand `doctor` target to print effective config (host, port, namespace, username)
- `tests/contract/test_config_single_authority.py` — New; contract tests for precedence

## Implementation Approach

### Phase 1 — Tests first (always first per project rules)

**New test file**: `tests/contract/test_config_single_authority.py`

- **Test 1.1**: YAML config drives connection (User Story 1, P1, Acceptance 1)
  - Create YAML with `database.iris.port: 51972` (non-default)
  - Pass to `ConfigurationManager`
  - Assert `get_database_config()["port"]` is `51972` (int)
  - Assert `ConnectionManager(config_manager).get_connection()` connects to port 51972
  - Mock DBAPI connect to verify call uses `port=51972`

- **Test 1.2**: Env var is None, YAML is non-default, no env (User Story 1, Acceptance 2)
  - No env vars set
  - YAML has port 51972
  - Assert fallback default 1972 **not** used; YAML 51972 used

- **Test 1.3**: Env > YAML precedence (User Story 2, P2, Acceptance 1)
  - Set `IRIS_PORT=51972` env var
  - YAML has port 1972
  - Assert env var wins: port is 51972

- **Test 1.4**: YAML > default precedence (User Story 2, Acceptance 2)
  - No env var
  - YAML has port 51972
  - Assert port 51972 (not fallback 1972)

- **Test 1.5**: No env, no YAML — default to 1972 (User Story 2, Acceptance 3)
  - No env vars, empty YAML config
  - Assert port defaults to 1972

- **Test 1.6**: `IRIS_USERNAME` and `IRIS_USER` dual support (Edge case, FR-005)
  - Set `IRIS_USER=alice` (legacy)
  - Assert username is "alice"
  - Set `IRIS_USERNAME=bob` (modern)
  - Assert username is "bob"
  - Set both: `IRIS_USERNAME=bob` should win

- **Test 1.7**: Invalid IRIS_PORT env var raises clear error (Edge case)
  - Set `IRIS_PORT=invalid_string`
  - Assert `ValueError` or `ValidationError` with clear message, not silent fallback

### Phase 2 — Implementation

#### Step 2.1: Fix defaults (FR-001)

**File**: `iris_vector_rag/config/default_config.yaml:6`

```yaml
# Before
port: "1974"

# After
port: 1972
```

**File**: `iris_vector_rag/config/manager.py:521`

```python
# Before
"port": 1974,

# After
"port": 1972,
```

#### Step 2.2: Add IRIS_USER fallback in manager.py (FR-005)

**File**: `iris_vector_rag/config/manager.py:537–558` (env_mappings loop)

Modify to check `IRIS_USERNAME` first, then fall back to `IRIS_USER`:

```python
# Before
env_mappings = {
    "IRIS_HOST": "host",
    "IRIS_PORT": "port",
    "IRIS_NAMESPACE": "namespace",
    "IRIS_USERNAME": "username",
    ...
}

for env_key, config_key in env_mappings.items():
    if env_key in os.environ:
        ...

# After (in manager.py, after default_config update from YAML):
# Handle IRIS_USERNAME vs IRIS_USER precedence
if "IRIS_USERNAME" in os.environ:
    default_config["username"] = os.environ["IRIS_USERNAME"]
elif "IRIS_USER" in os.environ:
    default_config["username"] = os.environ["IRIS_USER"]

# Then apply other env mappings (host, port, namespace, password, driver_path)
env_mappings = {
    "IRIS_HOST": "host",
    "IRIS_PORT": "port",
    "IRIS_NAMESPACE": "namespace",
    "IRIS_PASSWORD": "password",
    "IRIS_DRIVER_PATH": "driver_path",
}

for env_key, config_key in env_mappings.items():
    if env_key in os.environ:
        value = os.environ[env_key]
        if config_key == "port":
            try:
                value = int(value)
            except ValueError:
                raise ValueError(
                    f"IRIS_PORT must be an integer, got: {os.environ[env_key]}"
                )
        default_config[config_key] = value
```

#### Step 2.3: Wire ConnectionManager to use ConfigurationManager (FR-002)

**File**: `iris_vector_rag/core/connection.py:55–63`

Replace direct call to `get_iris_connection()` with call through `self.config_manager`:

```python
# Before
def get_connection(self, backend_name: str = "iris"):
    if backend_name != "iris":
        raise ValueError(f"Unsupported database backend: {backend_name}")

    from iris_vector_rag.common.iris_connection import get_iris_connection
    return get_iris_connection()

# After
def get_connection(self, backend_name: str = "iris"):
    if backend_name != "iris":
        raise ValueError(f"Unsupported database backend: {backend_name}")

    from iris_vector_rag.common.iris_connection import get_iris_connection

    db_config = self.config_manager.get_database_config()
    return get_iris_connection(
        host=db_config.get("host"),
        port=db_config.get("port"),
        namespace=db_config.get("namespace"),
        username=db_config.get("username"),
        password=db_config.get("password"),
    )
```

**Rationale**: `get_iris_connection()` already accepts explicit params (FR-003 already done
per iris_connection.py:225–238). Now `ConnectionManager` passes them explicitly instead
of relying on env-var fallback inside `get_iris_connection()`.

#### Step 2.4: Expand `make doctor` target (FR-006)

**File**: `Makefile` (existing doctor target)

Replace simple import check with effective config print:

```makefile
# Before
doctor: ## Verify environment and imports
 python -c "from iris_vector_rag import create_pipeline; print('✓ Import OK')"

# After
doctor: ## Show effective IRIS connection config
 python -c "\
from iris_vector_rag.config.manager import ConfigurationManager; \
cm = ConfigurationManager(); \
cfg = cm.get_database_config(); \
print('IRIS Configuration (Effective):'); \
print('  Host:       ' + str(cfg.get('host'))); \
print('  Port:       ' + str(cfg.get('port'))); \
print('  Namespace:  ' + str(cfg.get('namespace'))); \
print('  Username:   ' + str(cfg.get('username'))); \
print('  (password hidden)'); \
"
```

#### Step 2.5: Add validation to ConnectionManager (optional hardening)

**File**: `iris_vector_rag/core/connection.py` (in `get_connection()`)

After retrieving config, validate types:

```python
db_config = self.config_manager.get_database_config()
port = db_config.get("port")
if not isinstance(port, int):
    raise TypeError(f"Database port must be int, got {type(port).__name__}")
```

## Risks & Constraints

- **Risk**: Existing code that calls `get_iris_connection()` directly (bypassing `ConnectionManager`)
  will still read env vars and use `get_iris_connection()` defaults (1972). This is acceptable
  for now (backward compatibility); the spec targets `ConnectionManager` users.
  **Mitigation**: Document that `ConnectionManager` is the canonical path; new code should
  use `create_pipeline()` which uses `ConnectionManager`.

- **Risk**: Port as string in YAML ("1972") vs integer (1972). Manager.py now casts to int
  (line 554), so YAML port string is safe.
  **Mitigation**: Add assertion in test that `get_database_config()["port"]` is int, not string.

- **Risk**: If a user sets `IRIS_PORT=invalid`, manager.py will now raise `ValueError` instead
  of silently falling back. This is correct per FR-007 (fail fast), but could break legacy
  scripts that relied on silent fallback.
  **Mitigation**: Document in release notes; clear error message helps debugging.

- **Risk**: `.env.example` already specifies `IRIS_PORT=1972`, so tests must mock env vars
  to test other values (use `monkeypatch` in pytest).

## Dependencies

- **Prerequisite**: All 219+ unit tests must pass before and after changes.
- **Blocks**: No downstream dependencies yet; this enables future single-authority work
  (e.g., secret management, multi-namespace support).
- **Related**: Feature 074 (configurable schema prefix) does not depend on this; schema
  prefix is orthogonal to connection params.

## Success Criteria (SC)

- SC-001: Unit test verifies YAML port 51972 (no env var) results in connection to 51972.
- SC-002: Grep confirms `default_config.yaml:6` and `manager.py:521` both use `1972`.
- SC-003: All 219+ existing tests pass post-change.
- SC-004: `make doctor` prints resolved host/port/namespace/username, exits 0.
- SC-005: Invalid `IRIS_PORT` env var raises clear `ValueError`, not silent fallback.
