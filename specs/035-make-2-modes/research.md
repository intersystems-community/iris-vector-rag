# Research: Configurable Test Backend Modes

**Date**: 2025-10-08
**Feature**: 035-make-2-modes
**Phase**: Phase 0 - Research & Outline

## Research Questions

### 1. iris-devtools API and Integration Patterns

**Decision**: Use iris-devtools as a required sibling dependency via Python path manipulation

**Rationale**:
- iris-devtools provides battle-tested IRIS infrastructure utilities
- Already implemented in ../iris-pgwire project with proven patterns
- Avoids duplicating container management, password reset, connection logic
- Provides standardized pytest fixtures for test isolation

**iris-devtools Key APIs** (from ../iris-devtools investigation):

```python
# Container Lifecycle
from iris_devtools.containers import IRISContainer

with IRISContainer.community() as iris:
    conn = iris.get_connection()  # Auto password reset, wait strategies

with IRISContainer.enterprise(license_key="...") as iris:
    conn = iris.get_connection()

# Testing Utilities
from iris_devtools.testing import iris_test_fixture

@pytest.fixture(scope="module")
def iris_db():
    return iris_test_fixture()  # Returns (connection, state)

# Connection Management
from iris_devtools.connections import get_connection
from iris_devtools.config import IRISConfig

config = IRISConfig(host="localhost", port=1972, namespace="USER")
conn = get_connection(config)  # DBAPI-first, JDBC fallback

# Monitoring & Performance
from iris_devtools.containers.performance import get_resource_metrics
from iris_devtools.containers.monitoring import configure_monitoring

metrics = get_resource_metrics(container)
configure_monitoring(container, policy=MonitoringPolicy.BASIC)
```

**Integration Pattern**:
```python
# In iris_rag/testing/iris_devtools_bridge.py
import sys
from pathlib import Path

# Add iris-devtools to path
IRIS_DEVTOOLS_PATH = Path(__file__).parent.parent.parent / "iris-devtools"
if not IRIS_DEVTOOLS_PATH.exists():
    raise ImportError(
        f"iris-devtools not found at {IRIS_DEVTOOLS_PATH}\n"
        "Required development dependency.\n"
        "Clone from: https://github.com/your-org/iris-devtools"
    )
sys.path.insert(0, str(IRIS_DEVTOOLS_PATH))

from iris_devtools.containers import IRISContainer
from iris_devtools.testing import iris_test_fixture
```

**Alternatives Considered**:
- PyPI package: Not yet published, sibling dependency more flexible during dev
- Git submodule: Adds complexity, path-based import simpler
- Inline duplication: Violates Constitution VII (Standardized Database Interfaces)

---

### 2. IRIS Edition Detection Methods

**Decision**: Use SQL query `SELECT $SYSTEM.License.LicenseType()` to detect edition

**Rationale**:
- Community Edition returns specific license type identifier
- Enterprise Edition returns different identifier based on license
- Direct SQL query works regardless of connection method (DBAPI/JDBC)
- No file system access needed (works in Docker containers)

**Implementation**:
```python
def detect_iris_edition(connection) -> IRISEdition:
    """
    Detect IRIS edition from active connection.

    Returns:
        IRISEdition.COMMUNITY or IRISEdition.ENTERPRISE

    Raises:
        EditionDetectionError: If detection fails
    """
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT $SYSTEM.License.LicenseType()")
        license_type = cursor.fetchone()[0]

        # Community Edition license types
        if "community" in license_type.lower():
            return IRISEdition.COMMUNITY

        # Enterprise/Licensed editions
        return IRISEdition.ENTERPRISE

    except Exception as e:
        raise EditionDetectionError(
            f"Failed to detect IRIS edition: {e}\n"
            "Ensure IRIS connection is active and accessible."
        ) from e
```

**Alternatives Considered**:
- Docker image name parsing: Unreliable (can run any image)
- License file detection: Requires file system access, fails in containers
- Connection limit testing: Destructive, could break existing connections
- Environment variable: User must set, defeats auto-detection purpose

**Testing Strategy**:
- Mock edition detection for unit tests
- Real detection against both Community and Enterprise containers in integration tests
- Validate error handling when detection fails

---

### 3. pytest Connection Pooling Behavior

**Decision**: Use pytest session-scoped fixture with connection limit enforced via threading.Semaphore

**Rationale**:
- pytest doesn't have built-in connection pooling - we control it
- Session-scoped fixture ensures single pool instance across all tests
- Semaphore provides thread-safe connection limiting
- Works with both sequential and parallel pytest execution

**Implementation**:
```python
# tests/conftest.py
import threading
from enum import Enum

class BackendMode(Enum):
    COMMUNITY = "community"
    ENTERPRISE = "enterprise"

class ConnectionPool:
    def __init__(self, mode: BackendMode):
        self.mode = mode
        if mode == BackendMode.COMMUNITY:
            self._semaphore = threading.Semaphore(1)  # Max 1 connection
        else:
            self._semaphore = threading.Semaphore(999)  # Effectively unlimited

    def acquire_connection(self, iris_config):
        """Acquire connection, blocking if limit reached."""
        self._semaphore.acquire()
        try:
            return get_iris_connection(iris_config)
        except:
            self._semaphore.release()
            raise

    def release_connection(self, conn):
        """Release connection back to pool."""
        conn.close()
        self._semaphore.release()

@pytest.fixture(scope="session")
def backend_mode():
    """Determine backend mode from config + env var."""
    env_mode = os.getenv("IRIS_BACKEND_MODE")
    if env_mode:
        return BackendMode(env_mode.lower())

    # Read from config file
    config = load_config()
    return BackendMode(config.get("backend_mode", "community"))

@pytest.fixture(scope="session")
def connection_pool(backend_mode):
    """Session-wide connection pool."""
    return ConnectionPool(backend_mode)

@pytest.fixture
def iris_connection(connection_pool, iris_config):
    """Per-test connection from pool."""
    conn = connection_pool.acquire_connection(iris_config)
    yield conn
    connection_pool.release_connection(conn)
```

**Alternatives Considered**:
- SQLAlchemy connection pool: Heavyweight, adds dependency
- Queue-based pool: More complex, Semaphore simpler
- No pooling (direct connections): Can't enforce limits
- pytest-xdist integration: Not needed, Semaphore works with any executor

**Testing Strategy**:
- Unit tests mock Semaphore behavior
- Integration tests verify 1-connection limit in community mode
- Integration tests verify parallel execution in enterprise mode
- Performance tests validate >95% license error prevention

---

### 4. Configuration Override Precedence

**Decision**: Environment variable > Config file > Default (COMMUNITY)

**Rationale**:
- Standard precedence order (most CI/CD systems use env vars)
- Allows per-developer overrides without modifying config file
- Explicit over implicit (env var is most explicit)
- Matches common practice (Docker, 12-factor apps)

**Implementation**:
```python
from pathlib import Path
import os
import yaml

def load_backend_configuration() -> BackendConfiguration:
    """
    Load backend configuration with precedence:
    1. IRIS_BACKEND_MODE environment variable
    2. backend_mode in config/backend_modes.yaml
    3. Default: COMMUNITY

    Raises:
        ConfigurationError: If invalid mode specified
    """
    # 1. Check environment variable (highest precedence)
    env_mode = os.getenv("IRIS_BACKEND_MODE")
    if env_mode:
        try:
            mode = BackendMode(env_mode.lower())
            logger.info(f"Backend mode from env var: {mode.value}")
            return BackendConfiguration(mode=mode, source="environment")
        except ValueError:
            raise ConfigurationError(
                f"Invalid IRIS_BACKEND_MODE: {env_mode}\n"
                "Valid values: community, enterprise"
            )

    # 2. Check config file
    config_path = Path(".specify/config/backend_modes.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            mode_str = config.get("backend_mode")
            if mode_str:
                mode = BackendMode(mode_str.lower())
                logger.info(f"Backend mode from config file: {mode.value}")
                return BackendConfiguration(mode=mode, source="config_file")

    # 3. Default to COMMUNITY (safest)
    logger.info("Backend mode defaulted to: community")
    return BackendConfiguration(mode=BackendMode.COMMUNITY, source="default")
```

**Configuration File Format** (`.specify/config/backend_modes.yaml`):
```yaml
# Backend mode configuration for IRIS test execution
# Options: community | enterprise
backend_mode: community

# Optional: Custom iris-devtools path (default: ../iris-devtools)
# iris_devtools_path: /custom/path/to/iris-devtools

# Optional: Override connection limits (for testing)
# max_connections:
#   community: 1
#   enterprise: null  # unlimited
```

**Environment Variable Usage**:
```bash
# Override to enterprise mode for single test run
IRIS_BACKEND_MODE=enterprise pytest tests/

# Set permanently in shell
export IRIS_BACKEND_MODE=community

# In CI/CD (GitHub Actions example)
env:
  IRIS_BACKEND_MODE: enterprise
```

**Alternatives Considered**:
- Config file > Env var: Less flexible, breaks CI/CD patterns
- Command-line flag: Requires pytest plugin, more complex
- Auto-detection only: No manual override, too rigid
- Multiple config files: Complexity without benefit

**Validation**:
- Config file schema validation via jsonschema
- Environment variable value validation (community|enterprise only)
- Clear error messages with examples when invalid

---

## Summary of Decisions

| Question | Decision | Key Benefit |
|----------|----------|-------------|
| iris-devtools Integration | Sibling dependency via path manipulation | Reuse battle-tested utilities, avoid duplication |
| Edition Detection | SQL query `$SYSTEM.License.LicenseType()` | Reliable, works in containers, no file access needed |
| Connection Pooling | pytest fixtures + threading.Semaphore | Thread-safe limits, works with any pytest executor |
| Configuration Precedence | Env var > Config file > Default | Standard practice, CI/CD friendly, explicit override |

## Constitutional Alignment

All research decisions align with constitution:

- **VII. Standardized Database Interfaces**: iris-devtools provides proven patterns
- **VI. Explicit Error Handling**: Clear errors for detection failures, invalid config
- **II. Pipeline Validation**: Auto-validation via edition detection, config validation
- **III. Test-Driven Development**: Test strategy defined for each component

## Next Steps

Phase 1 artifacts to generate based on this research:

1. **data-model.md**: Define BackendMode, BackendConfiguration, IRISEdition entities
2. **contracts/**: API contracts for configuration, edition detection, iris-devtools bridge
3. **Contract tests**: Test files for each contract (failing initially per TDD)
4. **quickstart.md**: Guide using research findings (env var examples, config file format)
5. **CLAUDE.md update**: Add iris-devtools patterns, backend mode configuration approach
