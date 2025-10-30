# Data Model: Configurable Test Backend Modes

**Date**: 2025-10-08
**Feature**: 035-make-2-modes
**Phase**: Phase 1 - Design & Contracts

## Overview

This data model defines the entities and relationships for the configurable test backend modes feature, enabling tests to execute reliably across IRIS Community and Enterprise editions with appropriate connection limits and execution strategies.

## Entity Definitions

### 1. BackendMode (Enum)

**Purpose**: Represents the two supported IRIS backend modes for testing.

**Values**:
- `COMMUNITY`: Community edition mode with strict connection limits
- `ENTERPRISE`: Enterprise edition mode with unlimited connections

**Validation Rules**:
- MUST be one of two defined values
- Case-insensitive when loaded from configuration ("community" → COMMUNITY)
- Invalid values MUST raise ConfigurationError with clear message

**Source**: Configuration file (`.specify/config/backend_modes.yaml`) or environment variable (`IRIS_BACKEND_MODE`)

**Python Representation**:
```python
from enum import Enum

class BackendMode(Enum):
    """IRIS backend mode for test execution."""
    COMMUNITY = "community"
    ENTERPRISE = "enterprise"

    @classmethod
    def from_string(cls, value: str) -> "BackendMode":
        """Parse mode from string (case-insensitive)."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ConfigurationError(
                f"Invalid backend mode: {value}\n"
                "Valid values: community, enterprise"
            )
```

---

### 2. IRISEdition (Enum)

**Purpose**: Represents the detected IRIS database edition at runtime.

**Values**:
- `COMMUNITY`: InterSystems IRIS Community Edition
- `ENTERPRISE`: InterSystems IRIS Enterprise/Licensed Edition

**Detection Method**: SQL query `SELECT $SYSTEM.License.LicenseType()`

**Validation Rules**:
- MUST be detectable from active IRIS connection
- Detection failure MUST raise EditionDetectionError
- MUST match configured BackendMode (enforced by validation)

**Relationships**:
- One-to-one with active IRIS connection
- Compared against BackendConfiguration.mode for mismatch detection

**Python Representation**:
```python
class IRISEdition(Enum):
    """Detected IRIS database edition."""
    COMMUNITY = "community"
    ENTERPRISE = "enterprise"

    @classmethod
    def detect(cls, connection) -> "IRISEdition":
        """
        Detect edition from active IRIS connection.

        Args:
            connection: Active IRIS database connection

        Returns:
            Detected IRISEdition

        Raises:
            EditionDetectionError: If detection fails
        """
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT $SYSTEM.License.LicenseType()")
            license_type = cursor.fetchone()[0]

            if "community" in license_type.lower():
                return cls.COMMUNITY

            return cls.ENTERPRISE

        except Exception as e:
            raise EditionDetectionError(
                f"Failed to detect IRIS edition: {e}"
            ) from e
```

---

### 3. BackendConfiguration

**Purpose**: Immutable configuration object holding backend mode settings and derived constraints.

**Fields**:
- `mode: BackendMode` - Selected backend mode (REQUIRED)
- `max_connections: int` - Maximum concurrent connections (derived from mode)
- `execution_strategy: ExecutionStrategy` - Test execution strategy (derived from mode)
- `iris_devtools_path: Path` - Path to iris-devtools dependency (default: ../iris-devtools)
- `source: ConfigSource` - Where configuration was loaded from (for logging)

**Validation Rules**:
- `mode` MUST be valid BackendMode
- `max_connections` MUST be 1 for COMMUNITY, unlimited (999) for ENTERPRISE
- `execution_strategy` MUST be SEQUENTIAL for COMMUNITY, PARALLEL for ENTERPRISE
- `iris_devtools_path` MUST exist and be accessible
- Configuration MUST be validated before test execution begins (FR-009)

**State Lifecycle**:
1. **Created**: Loaded from config file + env var at test session start
2. **Validated**: Edition detection performed, match validated
3. **Immutable**: Cannot be changed mid-test-session
4. **Released**: Test session ends

**Relationships**:
- Contains one BackendMode
- References one iris-devtools installation (via path)
- Validated against one detected IRISEdition

**Python Representation**:
```python
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

class ExecutionStrategy(Enum):
    """Test execution strategy."""
    SEQUENTIAL = "sequential"  # One test at a time
    PARALLEL = "parallel"      # Multiple tests concurrently

class ConfigSource(Enum):
    """Source of configuration value."""
    ENVIRONMENT = "environment"  # IRIS_BACKEND_MODE env var
    CONFIG_FILE = "config_file"  # backend_modes.yaml
    DEFAULT = "default"          # Hardcoded default

@dataclass(frozen=True)  # Immutable
class BackendConfiguration:
    """Immutable backend mode configuration."""
    mode: BackendMode
    source: ConfigSource
    iris_devtools_path: Path = Path("../iris-devtools")

    @property
    def max_connections(self) -> int:
        """Maximum concurrent database connections."""
        return 1 if self.mode == BackendMode.COMMUNITY else 999

    @property
    def execution_strategy(self) -> ExecutionStrategy:
        """Test execution strategy."""
        return (
            ExecutionStrategy.SEQUENTIAL
            if self.mode == BackendMode.COMMUNITY
            else ExecutionStrategy.PARALLEL
        )

    def validate(self, detected_edition: IRISEdition) -> None:
        """
        Validate configuration against detected IRIS edition.

        Args:
            detected_edition: Edition detected from IRIS connection

        Raises:
            EditionMismatchError: If mode doesn't match edition
        """
        expected_edition = (
            IRISEdition.COMMUNITY
            if self.mode == BackendMode.COMMUNITY
            else IRISEdition.ENTERPRISE
        )

        if detected_edition != expected_edition:
            raise EditionMismatchError(
                f"Backend mode '{self.mode.value}' does not match "
                f"detected IRIS edition '{detected_edition.value}'\n"
                f"Fix: Set IRIS_BACKEND_MODE={detected_edition.value} "
                f"or update config file"
            )

        if not self.iris_devtools_path.exists():
            raise IrisDevtoolsMissingError(
                f"iris-devtools not found at {self.iris_devtools_path}\n"
                "Required development dependency.\n"
                "Clone from: https://github.com/your-org/iris-devtools"
            )
```

---

### 4. ConnectionPool

**Purpose**: Manages database connections with mode-aware limits using thread-safe semaphore.

**Fields**:
- `mode: BackendMode` - Backend mode determining connection limit
- `_semaphore: threading.Semaphore` - Thread-safe connection counter
- `_active_connections: Dict[int, Connection]` - Tracking active connections

**Operations**:
- `acquire_connection(config) -> Connection` - Get connection, blocking if limit reached
- `release_connection(connection) -> None` - Return connection to pool
- `get_active_count() -> int` - Current number of active connections (for monitoring)

**Behavior**:
- Community mode: Semaphore(1) - blocks when 1 connection active
- Enterprise mode: Semaphore(999) - effectively unlimited
- Thread-safe: Multiple tests can safely acquire/release concurrently
- Timeout: acquire_connection has 30s timeout to prevent deadlock

**Validation Rules**:
- Active connections MUST NOT exceed max_connections for mode
- MUST block (not fail) when limit reached in community mode
- MUST log warning if blocking occurs (indicates potential test parallelism issue)

**Python Representation**:
```python
import threading
from typing import Dict

class ConnectionPool:
    """Thread-safe connection pool with mode-aware limits."""

    def __init__(self, config: BackendConfiguration):
        self.config = config
        self._semaphore = threading.Semaphore(config.max_connections)
        self._active_connections: Dict[int, Connection] = {}
        self._lock = threading.Lock()

    def acquire_connection(self, iris_config, timeout: float = 30.0) -> Connection:
        """
        Acquire connection from pool.

        Args:
            iris_config: IRIS connection configuration
            timeout: Maximum seconds to wait for connection

        Returns:
            Active database connection

        Raises:
            ConnectionPoolTimeout: If timeout exceeded
            ConnectionError: If connection fails
        """
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise ConnectionPoolTimeout(
                f"Connection pool timeout after {timeout}s\n"
                f"Mode: {self.config.mode.value} "
                f"(max {self.config.max_connections} connections)\n"
                "Possible cause: Test parallelism exceeds connection limit"
            )

        try:
            conn = get_iris_connection(iris_config)
            with self._lock:
                self._active_connections[id(conn)] = conn
            return conn
        except:
            self._semaphore.release()
            raise

    def release_connection(self, connection: Connection) -> None:
        """Release connection back to pool."""
        with self._lock:
            self._active_connections.pop(id(connection), None)
        connection.close()
        self._semaphore.release()

    def get_active_count(self) -> int:
        """Get current number of active connections."""
        with self._lock:
            return len(self._active_connections)
```

---

### 5. IrisDevToolsBridge

**Purpose**: Adapter for iris-devtools functionality, providing container lifecycle and database state management.

**Fields**:
- `devtools_path: Path` - Path to iris-devtools installation
- `_container: Optional[IRISContainer]` - Active container reference
- `config: BackendConfiguration` - Backend configuration

**Operations**:
- `start_container(edition: IRISEdition) -> IRISContainer` - Start appropriate IRIS container
- `stop_container() -> None` - Stop active container
- `reset_schema(connection) -> None` - Reset database schema to clean state
- `validate_connection(connection) -> bool` - Validate connection health
- `check_health() -> Dict[str, Any]` - Get container health metrics

**Validation Rules**:
- iris-devtools MUST be importable from devtools_path
- Container start MUST match configured mode (Community/Enterprise image)
- Health check MUST return True before allowing tests to proceed

**Relationships**:
- Uses BackendConfiguration to determine container type
- Manages zero or one IRISContainer at a time
- Delegates to iris-devtools.containers.IRISContainer for actual operations

**Python Representation**:
```python
import sys
from pathlib import Path
from typing import Optional, Dict, Any

class IrisDevToolsBridge:
    """Bridge to iris-devtools container and state management."""

    def __init__(self, config: BackendConfiguration):
        self.config = config
        self.devtools_path = config.iris_devtools_path
        self._container: Optional[Any] = None  # IRISContainer
        self._import_devtools()

    def _import_devtools(self) -> None:
        """Import iris-devtools package."""
        if not self.devtools_path.exists():
            raise IrisDevtoolsMissingError(
                f"iris-devtools not found at {self.devtools_path}"
            )

        sys.path.insert(0, str(self.devtools_path))
        try:
            from iris_devtools.containers import IRISContainer
            self._IRISContainer = IRISContainer
        except ImportError as e:
            raise IrisDevtoolsImportError(
                f"Failed to import iris-devtools: {e}"
            ) from e

    def start_container(self, edition: IRISEdition) -> Any:
        """Start IRIS container matching edition."""
        if edition == IRISEdition.COMMUNITY:
            self._container = self._IRISContainer.community()
        else:
            self._container = self._IRISContainer.enterprise()

        self._container.start()
        return self._container

    def reset_schema(self, connection) -> None:
        """Reset database schema to clean state."""
        # Use iris-devtools schema reset utilities
        from iris_devtools.testing import reset_schema
        reset_schema(connection)

    def validate_connection(self, connection) -> bool:
        """Validate connection health."""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            return cursor.fetchone()[0] == 1
        except:
            return False

    def check_health(self) -> Dict[str, Any]:
        """Get container health metrics."""
        if not self._container:
            return {"status": "no_container"}

        from iris_devtools.containers.performance import get_resource_metrics
        return get_resource_metrics(self._container)
```

---

## Relationships Diagram

```
┌─────────────────────┐
│ BackendConfiguration│
├─────────────────────┤
│ + mode              │◄────┐
│ + max_connections   │     │
│ + execution_strategy│     │
│ + iris_devtools_path│     │
└─────────────────────┘     │
           │                │
           │ contains       │ uses
           ▼                │
    ┌──────────┐            │
    │BackendMode│            │
    ├──────────┤            │
    │ COMMUNITY│            │
    │ENTERPRISE│            │
    └──────────┘            │
                            │
┌─────────────┐             │
│ IRISEdition │             │
├─────────────┤             │
│ COMMUNITY   │             │
│ ENTERPRISE  │             │
└─────────────┘             │
       │                    │
       │ detected from      │
       ▼                    │
┌─────────────────┐         │
│ IRIS Connection │         │
└─────────────────┘         │
       │                    │
       │ managed by         │
       ▼                    │
┌──────────────────┐        │
│ ConnectionPool   │────────┘
├──────────────────┤
│ + acquire()      │
│ + release()      │
│ + get_active()   │
└──────────────────┘
       │
       │ uses
       ▼
┌────────────────────┐
│IrisDevToolsBridge  │
├────────────────────┤
│ + start_container()│
│ + stop_container() │
│ + reset_schema()   │
│ + validate_conn()  │
│ + check_health()   │
└────────────────────┘
       │
       │ delegates to
       ▼
┌────────────────────┐
│ iris-devtools      │
│ (external package) │
└────────────────────┘
```

## Error Hierarchy

```
BackendModeError (base)
├── ConfigurationError
│   ├── InvalidModeError
│   └── ConfigFileError
├── EditionDetectionError
│   └── ConnectionRequiredError
├── EditionMismatchError
├── IrisDevtoolsError
│   ├── IrisDevtoolsMissingError
│   └── IrisDevtoolsImportError
└── ConnectionPoolError
    ├── ConnectionPoolTimeout
    └── ConnectionLimitExceeded
```

All errors MUST include:
- Clear description of what failed
- Actionable fix instructions
- Context (current mode, detected edition, etc.)

## State Transitions

### Backend Configuration Lifecycle

```
[Test Session Start]
        │
        ▼
   Load Config
   (env var > file > default)
        │
        ▼
   Create BackendConfiguration
   (immutable from this point)
        │
        ▼
   Detect IRIS Edition
   (SQL query)
        │
        ▼
   Validate Match
   (mode == edition?)
        │
        ├─No──► FAIL (EditionMismatchError)
        │
        ▼
   Initialize ConnectionPool
   (with mode-specific limits)
        │
        ▼
   [Tests Execute]
        │
        ▼
   [Test Session End]
```

### Connection Lifecycle

```
[Test Starts]
        │
        ▼
   Request Connection
   (from fixture)
        │
        ▼
   Acquire from Pool
   (may block if limit reached)
        │
        ▼
   [Test Uses Connection]
        │
        ▼
   [Test Ends]
        │
        ▼
   Release to Pool
   (semaphore.release())
```

## Validation Matrix

| Entity | Validation Rule | When Checked | Error Type |
|--------|----------------|--------------|------------|
| BackendMode | Value in {COMMUNITY, ENTERPRISE} | Config load | ConfigurationError |
| IRISEdition | Detectable from connection | Pre-test | EditionDetectionError |
| BackendConfiguration | mode matches detected edition | Pre-test | EditionMismatchError |
| BackendConfiguration | iris_devtools_path exists | Config load | IrisDevtoolsMissingError |
| ConnectionPool | Active ≤ max_connections | Per acquire | ConnectionPoolTimeout |
| IrisDevToolsBridge | iris-devtools importable | Init | IrisDevtoolsImportError |

## Constitutional Alignment

- **I. Framework-First**: All entities reusable, no app-specific logic
- **II. Pipeline Validation**: BackendConfiguration.validate() enforces requirements
- **VI. Explicit Error Handling**: Complete error hierarchy with actionable messages
- **VII. Standardized Interfaces**: IrisDevToolsBridge uses proven iris-devtools patterns
