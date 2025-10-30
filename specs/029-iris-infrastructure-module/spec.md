# Feature 029: Reusable IRIS Infrastructure Module

**Status**: Planning
**Created**: 2025-10-05
**Priority**: High
**Epic**: Testing Infrastructure & Developer Experience

## Executive Summary

Extract IRIS database infrastructure management into a reusable, standalone Python package that can be used across all IRIS projects. This module will consolidate hard-won learnings about Docker, IRIS database configuration, password management, connection pooling, and testing infrastructure into a battle-tested, constitutionalized library.

## Problem Statement

### Current Pain Points

1. **Duplicated Infrastructure Code**: Every IRIS project requires copying connection managers, Docker configuration, password reset utilities
2. **Blind Alleys Revisited**: Developers repeatedly encounter solved problems (password expiration, port conflicts, JDBC vs DBAPI)
3. **Inconsistent Testing**: Each project implements testing infrastructure differently, with varying reliability
4. **Knowledge Fragmentation**: Best practices scattered across multiple projects, not codified
5. **Onboarding Friction**: New IRIS projects require extensive setup and debugging

### Success Criteria

- ✅ Single `pip install iris-devtools` provides all IRIS infrastructure
- ✅ Zero-configuration IRIS container management for testing
- ✅ Automatic password reset and connection recovery
- ✅ Seamless integration with pytest and testcontainers
- ✅ Medical-grade reliability based on battle-tested code
- ✅ Comprehensive documentation of all learnings

## Research Findings

### testcontainers-iris-python (caretdev)

**Repository**: https://github.com/caretdev/testcontainers-iris-python
**Status**: Active (v1.2.2, Feb 2025)
**License**: MIT

**Key Features**:
- Extends testcontainers-python for IRIS
- Automatic container lifecycle management
- SQLAlchemy integration via `get_connection_url()`
- Enterprise edition support with license key
- Custom namespace and user creation
- Automatic cleanup via Ryuk sidecar

**Example Usage**:
```python
from testcontainers.iris import IRISContainer
import sqlalchemy

iris_container = IRISContainer("intersystemsdc/iris-community:latest")
with iris_container as iris:
    engine = sqlalchemy.create_engine(iris.get_connection_url())
    with engine.begin() as connection:
        result = connection.execute(sqlalchemy.text("select $zversion"))
```

**Enterprise Configuration**:
```python
iris_container = IRISContainer(
    "containers.intersystems.com/intersystems/iris:2023.3",
    license_key="/full/path/to/iris.key",
    username="testuser",
    password="testpass",
    namespace="TEST",
)
```

### Best Practices from Research

1. **Isolation via Containers**: Each test suite gets dedicated IRIS instance
2. **Automatic Cleanup**: Ryuk sidecar ensures no orphaned containers
3. **Wait Strategies**: Container readiness detection before test execution
4. **Fixture-Based Setup**: pytest fixtures with finalizers for guaranteed cleanup
5. **Scope Management**: Module-scoped fixtures for performance, function-scoped for isolation
6. **Connection URL Abstraction**: Hide complexity behind simple `get_connection_url()`

## Proposed Architecture

### Module Name: `iris-devtools`

A comprehensive Python package providing:

```
iris-devtools/
├── iris_devtools/
│   ├── __init__.py
│   ├── containers/           # Testcontainers integration
│   │   ├── __init__.py
│   │   ├── iris_container.py      # Extended IRISContainer with auto-remediation
│   │   ├── wait_strategies.py     # Custom wait strategies
│   │   └── fixtures.py            # Pytest fixtures
│   ├── connections/          # Connection management
│   │   ├── __init__.py
│   │   ├── manager.py             # Unified connection manager (DBAPI/JDBC)
│   │   ├── pool.py                # Connection pooling
│   │   └── recovery.py            # Automatic password reset & recovery
│   ├── testing/              # Testing utilities
│   │   ├── __init__.py
│   │   ├── preflight.py           # Pre-flight checks
│   │   ├── schema_manager.py      # Schema validation & reset
│   │   ├── cleanup.py             # Test data cleanup
│   │   └── state.py               # Test state management
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   ├── defaults.py            # Default configurations
│   │   └── discovery.py           # Environment detection
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── docker_helpers.py      # Docker inspection utilities
│       └── diagnostics.py         # Diagnostic tools
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── quickstart.md
│   ├── best-practices.md
│   ├── troubleshooting.md
│   └── learnings/             # Codified learnings
│       ├── password-management.md
│       ├── connection-strategies.md
│       └── docker-pitfalls.md
├── examples/
│   ├── basic_usage.py
│   ├── pytest_integration.py
│   └── enterprise_setup.py
├── pyproject.toml
├── README.md
├── CONSTITUTION.md            # Module principles
└── docs/docs/docs/CHANGELOG.md
```

## Core Components

### 1. IRISContainer (Enhanced)

Extends `testcontainers-iris-python` with automatic remediation:

```python
from iris_devtools.containers import IRISContainer

# Zero-config usage
with IRISContainer.community() as iris:
    conn = iris.get_connection()
    # Connection automatically handles password resets

# Enterprise usage with auto-license discovery
with IRISContainer.enterprise(
    license_key_path="~/.iris/iris.key",  # Auto-discovers
    namespace="MYAPP"
) as iris:
    conn = iris.get_connection()
```

**Features**:
- Automatic password reset on "Password change required"
- Smart port allocation (avoids conflicts)
- Health check wait strategies
- Schema auto-initialization
- Durable storage for persistence scenarios
- Automatic cleanup even on crashes

### 2. Connection Manager (Unified)

Battle-tested connection management:

```python
from iris_devtools.connections import IRISConnectionManager

# Auto-detects best connection method (DBAPI -> JDBC -> fallback)
manager = IRISConnectionManager.from_env()

# Automatic recovery
with manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    # Password resets happen transparently
```

**Features**:
- DBAPI-first with JDBC fallback
- Automatic password reset detection/remediation
- Connection pooling with health checks
- Environment-based configuration
- Mock mode for unit tests without database

### 3. Testing Utilities

Pytest-ready testing infrastructure:

```python
# pytest conftest.py
from iris_devtools.testing import iris_test_fixture

@pytest.fixture(scope="module")
def iris_db():
    """Provides clean IRIS instance for tests."""
    return iris_test_fixture(
        schema_validator=True,    # Auto-validate schema
        auto_cleanup=True,        # Clean test data
        reset_on_mismatch=True,   # Auto-reset schema if stale
    )

def test_my_feature(iris_db):
    conn, state = iris_db
    # Test with guaranteed clean state
```

**Features**:
- Schema validation & auto-reset
- Pre-flight checks with auto-remediation
- Test data isolation via test_run_id
- Automatic cleanup (even on failures)
- Performance monitoring (warns on slow tests)

### 4. Configuration Discovery

Smart environment detection:

```python
from iris_devtools.config import IRISConfig

# Auto-discovers from:
# 1. Environment variables
# 2. .env files
# 3. Docker container inspection
# 4. testcontainers instance
config = IRISConfig.auto_discover()

# Or explicit
config = IRISConfig(
    host="localhost",
    port=1972,
    namespace="USER",
    username="_SYSTEM",
    password="SYS",
)
```

## Codified Learnings

### CONSTITUTION.md Principles

1. **DBAPI First**: Always prefer intersystems-irispython over JDBC
2. **Automatic Recovery**: Password resets must be automatic and transparent
3. **Isolation by Default**: Tests get dedicated containers unless explicitly shared
4. **Fail Fast with Guidance**: Clear error messages with remediation steps
5. **Zero Config Viable**: `pip install && pytest` should work
6. **Enterprise Ready**: Support both community and enterprise editions
7. **Medical Grade**: All code battle-tested in production scenarios

### Documented Learnings

**docs/learnings/password-management.md**:
- Why passwords expire in Docker containers
- Automatic reset via docker exec
- UnExpireUserPasswords in startup scripts
- Environment variable synchronization

**docs/learnings/connection-strategies.md**:
- DBAPI vs JDBC: when each is appropriate
- Port discovery algorithms
- Connection pooling best practices
- Error recovery patterns

**docs/learnings/docker-pitfalls.md**:
- Port conflicts and resolution
- Volume mounting for durable storage
- Network isolation issues
- Container cleanup strategies
- Resource limits and OOM issues

## Implementation Plan

### Phase 1: Core Extraction (Week 1)

**T001**: Create `iris-devtools` package structure
- Setup pyproject.toml with dependencies
- Create module skeleton
- Add testcontainers-iris-python as dependency

**T002**: Extract connection manager
- Port IRISConnectionManager from rag-templates
- Add password reset integration
- Implement auto-recovery logic
- Add comprehensive tests

**T003**: Extract password reset utility
- Port IRISPasswordResetHandler
- Add Docker detection logic
- Document all reset methods
- Add integration tests

### Phase 2: Testcontainers Integration (Week 1-2)

**T004**: Enhanced IRISContainer
- Extend caretdev's IRISContainer
- Add automatic password remediation
- Add schema initialization hooks
- Add health check wait strategies

**T005**: pytest fixtures
- Create reusable fixtures
- Add scope management
- Implement cleanup handlers
- Document usage patterns

**T006**: Schema management
- Port SchemaValidator
- Port SchemaResetter
- Add auto-reset on mismatch
- Add performance monitoring

### Phase 3: Testing Infrastructure (Week 2)

**T007**: Pre-flight checks
- Port PreflightChecker
- Add auto-remediation
- Add diagnostic reporting
- Add integration with pytest

**T008**: Test isolation utilities
- Port TestDatabaseState
- Port DatabaseCleanupHandler
- Add test_run_id tracking
- Add cleanup verification

**T009**: Configuration system
- Implement auto-discovery
- Add .env support
- Add Docker inspection
- Add validation

### Phase 4: Documentation (Week 2-3)

**T010**: Quickstart guide
- Basic usage examples
- pytest integration
- Common patterns
- Troubleshooting

**T011**: Best practices guide
- Fixture scope management
- Performance optimization
- Enterprise configuration
- Security considerations

**T012**: Learnings documentation
- Password management deep-dive
- Connection strategy guide
- Docker pitfalls guide
- Case studies from rag-templates

### Phase 5: Publishing (Week 3)

**T013**: Package for PyPI
- Setup PyPI configuration
- Add CI/CD for releases
- Create release notes
- Tag v1.0.0

**T014**: Integration back into rag-templates
- Replace local implementations
- Update imports
- Verify all tests pass
- Document migration

## Dependencies

### External Packages
- `testcontainers-iris-python>=1.2.2` (caretdev)
- `testcontainers>=4.0.0`
- `intersystems-irispython>=3.0.0` (optional, for DBAPI)
- `jaydebeapi>=1.2.3` (optional, for JDBC)
- `docker>=6.0.0`
- `pytest>=8.0.0`
- `python-dotenv>=1.0.0`

### Development Dependencies
- `pytest-cov`
- `black`
- `isort`
- `mypy`
- `sphinx` (for docs)

## Success Metrics

### Adoption Metrics
- ✅ Used in 3+ IRIS projects within 6 months
- ✅ 50+ PyPI downloads per month
- ✅ Zero unresolved critical issues

### Quality Metrics
- ✅ 95%+ test coverage
- ✅ 100% of documented learnings codified
- ✅ <5 second container startup overhead
- ✅ 100% automatic password reset success rate

### Developer Experience
- ✅ New project setup in <10 minutes
- ✅ Zero configuration for basic usage
- ✅ Clear error messages with remediation
- ✅ Comprehensive documentation

## Migration Path

### For rag-templates

1. **Install iris-devtools**: `pip install iris-devtools`
2. **Update imports**:
   ```python
   # OLD
   from common.iris_connection_manager import get_iris_connection
   from tests.utils.iris_password_reset import IRISPasswordResetHandler

   # NEW
   from iris_devtools.connections import get_iris_connection
   from iris_devtools.connections.recovery import IRISPasswordResetHandler
   ```
3. **Replace fixtures**:
   ```python
   # OLD
   from tests.conftest import database_with_clean_schema

   # NEW
   from iris_devtools.testing import iris_test_fixture as database_with_clean_schema
   ```
4. **Update conftest.py**: Use iris-devtools fixtures
5. **Remove duplicated code**: Delete local implementations
6. **Verify tests**: Ensure all 771 tests pass

### For New Projects

```bash
# 1. Install
pip install iris-devtools

# 2. Create conftest.py
cat > conftest.py << 'EOF'
from iris_devtools.testing import iris_test_fixture
import pytest

@pytest.fixture(scope="module")
def iris_db():
    return iris_test_fixture()
EOF

# 3. Write tests
cat > test_example.py << 'EOF'
def test_basic_query(iris_db):
    conn, state = iris_db
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    assert cursor.fetchone()[0] == 1
EOF

# 4. Run tests
pytest
```

## Risks & Mitigation

### Risk: Breaking Changes in testcontainers-iris-python
**Mitigation**: Version pinning, comprehensive integration tests, maintain fork if needed

### Risk: IRIS Version Compatibility
**Mitigation**: Test against multiple IRIS versions (2023.x, 2024.x), document supported versions

### Risk: Platform-Specific Issues (Docker on Windows/Mac)
**Mitigation**: CI/CD on all platforms, document platform quirks, fallback strategies

### Risk: Enterprise License Management
**Mitigation**: Clear documentation, auto-discovery, support for multiple license sources

## Future Enhancements

- **IRIS Cloud Integration**: Support for IRIS Cloud instances
- **Multi-Container Orchestration**: IRIS + Mirror setups
- **Performance Profiling**: Built-in query profiling tools
- **Schema Migrations**: Automated migration tooling
- **Monitoring Hooks**: Integration with observability platforms
- **Web UI**: Dashboard for managing test containers

## References

- [testcontainers-iris-python](https://github.com/caretdev/testcontainers-iris-python)
- [testcontainers-python docs](https://testcontainers-python.readthedocs.io/)
- [InterSystems IRIS Docker Guide](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=ADOCK)
- [rag-templates Feature 026](specs/026-fix-critical-issues/) (Testing Framework)
- [rag-templates Feature 028](specs/028-obviously-these-failures/) (Test Infrastructure Resilience)

## Acceptance Criteria

- [ ] Package published to PyPI as `iris-devtools`
- [ ] Comprehensive documentation at ReadTheDocs
- [ ] 95%+ test coverage
- [ ] All learnings documented in `docs/learnings/`
- [ ] Example projects in `examples/`
- [ ] rag-templates successfully migrated
- [ ] Zero configuration quickstart works
- [ ] Automatic password reset 100% success rate
- [ ] Enterprise edition tested with real license
- [ ] CI/CD validates on Linux/Mac/Windows
